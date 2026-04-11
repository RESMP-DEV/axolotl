"""
Ax + BoTorch hyperparameter sweep for Axolotl LoRA fine-tuning.

Designed for Gemma4-31B on H100 (sm90a) but configurable for any model.
Default sweep target:
    - Flex (flex_attention: true + torch_compile: true)

Each trial runs a mini training run (max_steps=60 by default ≈ 5-8 min on
H100 with ZeRO-3 + LoRA) and returns the final train loss as the
minimisation objective.

Acquisition function:
    Uses qNoisyExpectedImprovement (qNEI) from BoTorch, which is the
    recommended choice for noisy objectives like training loss.  Standard
    Expected Improvement (EI) assumes noise-free observations and can be
    overly exploitative when loss varies across batches.

Usage:
    export HF_TOKEN=hf_...

    # Run sweep (15 trials by default):
    python examples/gemma4/sweep.py

    # Change trial count:
    python examples/gemma4/sweep.py --n_trials 10

    # Custom base config:
    python examples/gemma4/sweep.py --config path/to/my-config.yml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "gemma4"

# ─── default base configs ─────────────────────────────────────────────────────
BASE_CONFIGS = {
    "flex": EXAMPLES_DIR / "gemma-31b-musica-flex.yml",
}

# ─── sweep parameter space ─────────────────────────────────────────────────────
PARAMETERS = [
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [3e-7, 3e-5],
        "log_scale": True,
        "value_type": "float",
    },
    {
        "name": "warmup_ratio",
        "type": "range",
        "bounds": [0.01, 0.12],
        "value_type": "float",
    },
    {
        "name": "lora_r",
        "type": "choice",
        "values": [32, 64, 128],
        "value_type": "int",
        "is_ordered": True,
    },
    {
        "name": "max_grad_norm",
        "type": "range",
        "bounds": [0.1, 1.0],
        "value_type": "float",
    },
    {
        "name": "lora_dropout",
        "type": "range",
        "bounds": [0.0, 0.1],
        "value_type": "float",
    },
]

# Number of Sobol quasi-random trials before BoTorch GP kicks in.
N_SOBOL = 5
# Steps per mini-trial.  60 steps ≈ 5-8 min on H100 w/ ZeRO-3 full-bf16 LoRA.
MAX_STEPS_PER_TRIAL = 60


# ─── helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def apply_trial_params(base_cfg: dict, params: dict, trial_idx: int, run_dir: Path) -> dict:
    """Return a copy of base_cfg with sweep params and mini-run overrides applied."""
    cfg = deepcopy(base_cfg)

    # Sweep parameters
    cfg["learning_rate"] = float(params["learning_rate"])
    cfg["warmup_ratio"] = float(params["warmup_ratio"])
    cfg["lora_r"] = int(params["lora_r"])
    cfg["lora_alpha"] = int(params["lora_r"])  # keep alpha == r (standard practice)
    cfg["max_grad_norm"] = float(params["max_grad_norm"])
    cfg["lora_dropout"] = float(params["lora_dropout"])

    # Mini-run overrides (turn off epoch-based training, use step count)
    cfg["max_steps"] = MAX_STEPS_PER_TRIAL
    cfg.pop("num_epochs", None)

    # Unique output dir per trial so checkpoints don't collide
    cfg["output_dir"] = str(run_dir / f"trial_{trial_idx:03d}")
    cfg["dataset_prepared_path"] = str(run_dir / "prepared_dataset")

    # Disable comet for sweep trials (reduce overhead); keep wandb if set.
    # Must clear comet_project_name too — Axolotl re-enables comet if it's set.
    cfg["use_comet"] = False
    cfg.pop("comet_project_name", None)

    # Disable torch_compile in sweep trials — flex_attention compiles its own
    # kernels internally, and whole-model torch.compile adds overhead per trial.
    cfg["torch_compile"] = False
    cfg.pop("torch_compile_backend", None)

    # Enable gradient checkpointing to avoid OOM with 31B LoRA on single H100.
    # use_reentrant=True is required for ZeRO-3 + LoRA compatibility.
    # NOTE: Gemma4 ZeRO-3 + LoRA + gradient_checkpointing has recomputation
    # issues (saved vs recomputed tensor shape mismatch). Disable for now
    # and rely on seq_len=2048 + micro_batch=1 to keep VRAM in check.
    cfg["gradient_checkpointing"] = False

    # Use ZeRO-3 with CPU parameter offloading for 31B on single H100.
    # Without offload, the 31B model + LoRA activations exhaust 80GB VRAM.
    cfg["deepspeed"] = "deepspeed_configs/zero3_bf16_cpuoffload_params.json"

    # Reduce sequence length for sweep trials to avoid OOM on the logits tensor.
    # 31B model with seq_len=8192 → vocab_size × 8192 logits = ~8GB per forward pass.
    # 2048 gives enough signal for HP comparison while fitting in VRAM.
    cfg["sequence_len"] = 2048

    # Reduce saves during sweep to save time
    cfg["saves_per_epoch"] = 0
    cfg.pop("save_total_limit", None)

    return cfg


def extract_final_loss(log_text: str) -> float | None:
    """Parse train/loss from axolotl/HF Trainer stdout.

    Axolotl prints loss like: {'loss': '35.12', 'grad_norm': '9.051', ...}
    Trainer also prints:     {'loss': 1.2345, 'learning_rate': ...}
    or:                      "Step X | loss: 1.2345"
    """
    # Axolotl format: 'loss': '35.12' (value in quotes)
    matches = re.findall(r"'loss'\s*:\s*'([\d.]+)'", log_text)
    if not matches:
        # HF Trainer format: 'loss': 1.2345 (value unquoted)
        matches = re.findall(r"'loss'\s*:\s*([\d.]+)", log_text)
    if not matches:
        matches = re.findall(r"loss['\"]?\s*:\s*([\d.]+)", log_text)
    if matches:
        return float(matches[-1])
    return None


def run_trial(config_path: Path, variant: str, params: dict, trial_idx: int, run_dir: Path) -> float:
    """
    Create a temp config, run `axolotl train`, extract loss.
    Returns the final loss (lower is better), or 999 on failure.
    """
    base_cfg = load_yaml(config_path)
    trial_cfg = apply_trial_params(base_cfg, params, trial_idx, run_dir)
    cfg_path = run_dir / f"trial_{trial_idx:03d}_config.yml"
    save_yaml(trial_cfg, cfg_path)

    log.info(
        f"[{variant}] Trial {trial_idx} | "
        f"lr={params['learning_rate']:.2e}  warmup={params['warmup_ratio']:.3f}  "
        f"r={params['lora_r']}  grad_norm={params['max_grad_norm']:.2f}  "
        f"dropout={params['lora_dropout']:.3f}"
    )

    env = os.environ.copy()
    # Ensure HF_TOKEN is propagated for the private dataset
    if "HF_TOKEN" not in env:
        log.warning("HF_TOKEN not set — private dataset access will fail.")

    # Make sure venv's bin/ is on PATH so accelerate/torchrun find the right python
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")

    cmd = [
        sys.executable, "-m", "axolotl.cli.main", "train", str(cfg_path),
    ]

    trial_log_path = run_dir / f"trial_{trial_idx:03d}.log"
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 60 min hard limit per trial
            env=env,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout + result.stderr
        with open(trial_log_path, "w") as f:
            f.write(output)

        elapsed = time.time() - t0
        loss = extract_final_loss(output)

        if result.returncode != 0 or loss is None:
            log.warning(
                f"[{variant}] Trial {trial_idx} FAILED (rc={result.returncode}, "
                f"loss_parsed={loss}, elapsed={elapsed:.0f}s). "
                f"See {trial_log_path}"
            )
            return 999.0

        log.info(f"[{variant}] Trial {trial_idx} done | loss={loss:.4f} | elapsed={elapsed:.0f}s")
        return loss

    except subprocess.TimeoutExpired:
        log.error(f"[{variant}] Trial {trial_idx} TIMED OUT after 60 min")
        return 999.0


# NOTE on acquisition function:
# Ax 1.2.1 auto-selects `qLogNoisyExpectedImprovement` (qLogNEI) for
# single-objective optimization via `choose_botorch_acqf_class()`.  This is
# the log-space variant of qNEI — the recommended choice for noisy objectives
# like training loss.  No explicit acquisition function override is needed.
#
# Refs:
#   - Ax source: ax/generators/torch/botorch_modular/utils.py → choose_botorch_acqf_class
#   - BoTorch acquisition: botorch.org/docs/acquisition
#   - LoRA HP sweep paper (arXiv 2602.11171): GP + EI-family standard


def run_sweep(
    variant: str,
    config_path: Path,
    n_trials: int,
    run_dir: Path,
    n_sobol: int = N_SOBOL,
) -> dict:
    """Run Ax+BoTorch sweep for one attention variant. Returns best parameters."""
    from ax.service.ax_client import AxClient, ObjectiveProperties

    log.info(f"\n{'='*60}")
    log.info(f"Starting sweep: variant={variant}, n_trials={n_trials}")
    log.info(f"Base config: {config_path}")
    log.info(f"Run dir: {run_dir}")
    log.info(f"Acquisition: Sobol ({n_sobol}) → BoTorch(qNEI)")
    log.info(f"{'='*60}")

    run_dir.mkdir(parents=True, exist_ok=True)

    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=f"sweep_{variant}",
        parameters=PARAMETERS,
        objectives={"train_loss": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={
            "num_initialization_trials": n_sobol,
            "use_batch_trials": False,
        },
    )

    results_log = []

    for i in range(n_trials):
        params, trial_index = ax_client.get_next_trial()
        loss = run_trial(config_path, variant, params, i, run_dir)
        ax_client.complete_trial(trial_index=trial_index, raw_data={"train_loss": (loss, None)})

        record = {"trial": i, "trial_index": trial_index, "params": params, "loss": loss}
        results_log.append(record)

        with open(run_dir / "results.json", "w") as f:
            json.dump(results_log, f, indent=2)

        df = ax_client.get_trials_data_frame()
        log.info(f"\n Trials so far:\n{df.to_string()}\n")

    best_result = ax_client.get_best_parameters()
    if best_result is None:
        log.warning("No successful trials found — no best parameters.")
        return {}

    best_params, (means, _) = best_result
    best_loss = means.get("train_loss", float("inf"))

    log.info(f"\n{'='*60}")
    log.info(f"[{variant}] SWEEP COMPLETE")
    log.info(f"Best loss:   {best_loss:.4f}")
    log.info(f"Best params: {json.dumps(best_params, indent=2)}")
    log.info(f"{'='*60}")

    # Save final best config
    base_cfg = load_yaml(config_path)
    best_cfg = deepcopy(base_cfg)
    best_cfg["learning_rate"] = float(best_params["learning_rate"])
    best_cfg["warmup_ratio"] = float(best_params["warmup_ratio"])
    best_cfg["lora_r"] = int(best_params["lora_r"])
    best_cfg["lora_alpha"] = int(best_params["lora_r"])
    best_cfg["max_grad_norm"] = float(best_params["max_grad_norm"])
    best_cfg["lora_dropout"] = float(best_params["lora_dropout"])
    best_cfg["output_dir"] = f"./outputs/v1-{variant}-best"
    best_cfg["dataset_prepared_path"] = f"./last_run_prepared_{variant}"

    best_cfg_path = config_path.parent / f"{config_path.stem}-best.yml"
    save_yaml(best_cfg, best_cfg_path)
    log.info(f"Best config written → {best_cfg_path}")

    ax_client.save_to_json_file(str(run_dir / "ax_experiment.json"))

    return best_params


def main():
    global MAX_STEPS_PER_TRIAL, BASE_CONFIGS

    parser = argparse.ArgumentParser(
        description="Ax+BoTorch LoRA hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to base YAML config (overrides the default flex config)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=N_SOBOL + 10,
        help=f"Number of trials per variant (default: {N_SOBOL + 10})",
    )
    parser.add_argument(
        "--n_sobol", type=int, default=N_SOBOL,
        help=f"Number of initial Sobol trials (default: {N_SOBOL})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS_PER_TRIAL,
        help=f"Max training steps per trial (default: {MAX_STEPS_PER_TRIAL})",
    )
    parser.add_argument(
        "--run_dir", type=Path, default=REPO_ROOT / "outputs" / "sweep",
        help="Base directory for sweep outputs and logs",
    )
    args = parser.parse_args()

    MAX_STEPS_PER_TRIAL = args.max_steps

    if args.config:
        BASE_CONFIGS = {"custom": args.config}
        variants = ["custom"]
    else:
        variants = ["flex"]

    all_best = {}

    for variant in variants:
        config_path = BASE_CONFIGS[variant]
        if not config_path.exists():
            log.error(f"Config not found: {config_path}")
            sys.exit(1)
        run_dir = args.run_dir / variant
        best = run_sweep(variant, config_path, args.n_trials, run_dir, n_sobol=args.n_sobol)
        all_best[variant] = best

    # ── summary ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SWEEP SUMMARY — best params per variant")
    log.info("=" * 60)
    for variant, params in all_best.items():
        log.info(f"\n[{variant}]")
        for k, v in params.items():
            log.info(f"  {k}: {v}")

    summary_path = args.run_dir / "sweep_summary.json"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_best, f, indent=2)
    log.info(f"\nSummary written → {summary_path}")

    for variant in variants:
        config_path = BASE_CONFIGS[variant]
        best_cfg = config_path.parent / f"{config_path.stem}-best.yml"
        if best_cfg.exists():
            log.info(f"Best config [{variant}] → {best_cfg}")


if __name__ == "__main__":
    main()
