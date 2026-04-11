# Axolotl

Fine-tuning framework for LLMs. Config-driven: every training run is defined by a single YAML file.

## Tech Stack

Python, PyTorch, HuggingFace Transformers, TRL, PEFT (LoRA/QLoRA), DeepSpeed, FSDP, vLLM (for GRPO generation).

## Commands

```bash
axolotl train config.yaml              # Train (single or multi-GPU, auto-detected)
axolotl preprocess config.yaml         # Tokenize dataset and validate config
axolotl preprocess config.yaml --debug # Inspect tokenized samples and label masking
axolotl inference config.yaml          # Interactive inference
axolotl merge-lora config.yaml         # Merge LoRA adapter into base model
axolotl vllm-serve config.yaml         # Start vLLM server for GRPO/EBFT training
axolotl fetch examples                 # Download example configs
axolotl agent-docs                     # Show agent-optimized docs (bundled with pip package)
axolotl agent-docs grpo                # Topic-specific agent reference
axolotl config-schema                  # Dump config JSON schema
```

## Training Methods

| Method | Config Key | When to Use |
|--------|-----------|-------------|
| SFT | *(default)* | Input-output pairs, instruction tuning |
| DPO/IPO | `rl: dpo` / `rl: ipo` | Paired preference data (chosen vs rejected) |
| KTO | `rl: kto` | Unpaired binary preference labels |
| ORPO | `rl: orpo` | Single-stage alignment, no ref model |
| GRPO | `rl: grpo` | RL with verifiable reward functions (math, code) |
| EBFT | `rl: ebft` | Feature-matching rewards from internal representations |

Agent-specific references:
- [docs/agents/sft.md](docs/agents/sft.md) — supervised fine-tuning
- [docs/agents/preference_tuning.md](docs/agents/preference_tuning.md) — DPO, IPO, KTO, ORPO, SimPO
- [docs/agents/grpo.md](docs/agents/grpo.md) — GRPO online RL with reward functions
- [docs/agents/reward_modelling.md](docs/agents/reward_modelling.md) — outcome and process reward models
- [docs/agents/pretraining.md](docs/agents/pretraining.md) — continual pretraining
- [docs/agents/model_architectures.md](docs/agents/model_architectures.md) — model-specific quirks (Gemma4, Qwen3.5 MoE, etc.)
- [docs/agents/new_model_support.md](docs/agents/new_model_support.md) — debugging and adding support for new model architectures

## Config Pattern

All training is config-driven. A YAML file specifies model, adapter, dataset(s), and hyperparameters:

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: lora                    # or qlora, or omit for full fine-tune
datasets:
  - path: my_dataset
    type: chat_template          # prompt strategy (see docs/dataset-formats/)
output_dir: ./outputs/lora-out
```

Config schema: `src/axolotl/utils/schemas/config.py` (AxolotlInputConfig).

## Project Structure

```
src/axolotl/
  cli/                           # CLI entry points (train, preprocess, inference, merge_lora, vllm_serve)
  core/
    builders/                    # TrainerBuilder classes (causal.py for SFT, rl.py for RLHF)
    trainers/                    # Trainer classes, mixins (optimizer, scheduler, packing)
      dpo/                       # DPO trainer and config
      grpo/                      # GRPO trainer and sampler
  loaders/                       # Model, tokenizer, adapter, processor loading
  prompt_strategies/             # Dataset format handlers (chat_template, alpaca, dpo/, kto/, orpo/)
  utils/schemas/                 # Pydantic config schemas (config, model, training, peft, trl, fsdp)
  integrations/                  # Plugins (liger, cut_cross_entropy, swanlab, nemo_gym)
  monkeypatch/                   # Runtime patches for HF transformers

examples/                        # Example YAML configs by model (llama-3/, qwen2/, mistral/, ebft/)
deepspeed_configs/               # DeepSpeed JSON configs (zero2, zero3)
docs/                            # Quarto documentation site
```

## Code Conventions

- Config-driven: features are toggled via YAML, not code changes
- Prompt strategies: `src/axolotl/prompt_strategies/` — each `type:` value maps to a function
- Plugin system: `plugins:` list in config loads integration modules
- Trainer mixins: `core/trainers/mixins/` for composable trainer behaviors
- Schemas: all config validation via Pydantic in `utils/schemas/`

## Key Documentation

- [Getting Started](docs/getting-started.qmd) — quickstart tutorial
- [Choosing a Method](docs/choosing_method.qmd) — SFT vs DPO vs GRPO decision guide
- [Config Reference](docs/config-reference.qmd) — all config options
- [Dataset Formats](docs/dataset-formats/) — chat_template, alpaca, input_output, completion
- [RLHF](docs/rlhf.qmd) — DPO, KTO, ORPO, GRPO, EBFT configs and dataset formats
- [GRPO Deep Dive](docs/grpo.qmd) — async training, custom rewards, scaling
- [vLLM Serving](docs/vllm_serving.qmd) — vLLM setup for GRPO/EBFT
- [Multi-GPU](docs/multi-gpu.qmd) — FSDP and DeepSpeed
- [Training Stability](docs/training_stability.qmd) — debugging loss, NaN, OOM
- [Debugging](docs/debugging.qmd) — VSCode setup, Docker debugging

## H100 (Hopper / sm90a) Footguns

Lessons learned from running Axolotl on NVIDIA H100 80GB HBM3. These will waste hours if you don't know about them.

### Attention Implementation

**FA2 fails on Gemma 4 / large head-dim models.**
Gemma 4 uses `global_head_dim=512`. FlashAttention 2 has a hard maximum of 256 head-dim; FA4 has 128. **FA3 also has a max of 256** (despite being the Hopper-native build). None of the FlashAttention variants support 512-dim heads. Use `flex_attention: true`.

**FA3 requires a source build — no pip wheel for Hopper.**
There is no pre-built FA3 wheel that works with PyTorch ≥ 2.6 on Hopper. Build from source:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention-src
cd /tmp/flash-attention-src && git submodule update --init csrc/cutlass
cd hopper && CUDA_HOME=/usr/local/cuda-13.1 \
  PATH="/usr/local/cuda-13.1/bin:$PATH" \
  MAX_JOBS=20 TORCH_CUDA_ARCH_LIST="9.0a" TORCH_DONT_CHECK_COMPILER_ABI=1 \
  python setup.py install
```
Expect ~40 min on 20 cores (291 CUTLASS kernel instantiations).

**nvcc version mismatch trips the build (PyTorch CUDA 12.8 vs system nvcc 13.1).**
PyTorch 2.8.0+cu128 was compiled against CUDA 12.8; system nvcc may be 13.1. torch's `_check_cuda_version` raises by default. Patch it to warn:
```python
# .venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py  line ~506
# Change:  raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
# To:      logger.warning(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
```
CCCL libcudacxx headers (`cuda/std/utility` etc.) live at
`/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/` — nvcc 13.1 finds them automatically; nvcc 12.x does **not**.

**FA3 egg doesn't propagate to `sys.path` unless torch is imported first.**
`from flash_attn_interface import flash_attn_func` works only after `import torch` is already executed (needed to resolve `libc10.so`). In training scripts this is a non-issue; in quick one-liners, always `import torch` first.

**FA4 (flash-attn-4) also fails on Gemma 4** — max head-dim 128, same problem as FA2.

### venv / uv

**Always use `uv` — never raw `pip`.** The venv lives at `/root/axolotl/.venv`. Install uv first then use `uv pip install`.

**After `setup.py install` the egg lands in site-packages but pip doesn't know about it.**
`uv pip show flash-attn-3` will return nothing. Use `python -c "import flash_attn_interface"` to verify.

### Mixed Precision & dtypes

**Default bf16 is correct for H100.** H100 natively supports bf16 at full tensor-core throughput. Never override to fp16 unless the model explicitly requires it (fp16 clips at ±65504; bf16 clips at ±3.4×10³⁸).

**Use `tf32: true` in Axolotl config.** Ampere+ (including H100) has TF32 math for matmuls; leaving it off halves throughput for free-precision operations.

### torch.compile on H100

**`torch_compile: true` + DeepSpeed ZeRO-3 can deadlock** on first compilation step. Symptoms: training hangs at step 0 with no error. Workaround: compile with `torch_compile_backend: inductor` and set `TORCH_COMPILE_DISABLE_CUDAGRAPHS=1` if using ZeRO-3.

**`torch_compile` re-traces on every new sequence length.** With variable-length inputs and packing disabled this causes repeated JIT overhead. Use `sample_packing: true` + fixed max seq len, or set `TORCH_DYNAMO_SYMBOLIC_SHAPES_SPECIALIZATION=0`.

### Gemma 4 Specific

**`freeze_mm_modules: true` is required for multimodal Gemma 4 SFT.** Without it the vision encoder gradients flow through and the run either OOMs or diverges on text-only data.

**`chat_template: gemma4` must be set explicitly.** The generic `default` template doesn't insert the Gemma 4 BOS/EOS separators correctly; loss is computed over incorrect spans.

**DeepSpeed ZeRO-3 is needed for 31B on a single 80 GB H100.** ZeRO-2 will OOM at LoRA rank ≥ 64 + batch > 1. Use `deepspeed_config: deepspeed_configs/zero3_bf16.json`.

### Dataset / Tokenization

**Run `axolotl preprocess config.yaml --debug` before every new config.** The debug flag shows actual token sequences and label masks. A common mistake is label-masking the entire sequence (all `-100`) due to a wrong `type:` value — the run trains but loss never decreases.

**Private HF Hub datasets require `HF_TOKEN` set in the shell before launch.** Axolotl doesn't surface a clear error; it silently fails with a 401 that looks like a dataset not found.

### Environment Inventory (this machine)

| Component | Version / Path |
|-----------|---------------|
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA driver | 590.48.01 |
| CUDA toolkit | 13.1 @ `/usr/local/cuda-13.1/` |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 (MUST match torch version — use `uv pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128`) |
| Python / venv | 3.10.12 @ `/root/axolotl/.venv` |
| Axolotl | 0.16.0.dev0 (editable) |
| FlashAttention 2 | 2.8.3 |
| FlashAttention 3 | 3.0.0 (source build, `/root/axolotl/.venv/.../flash_attn_3-3.0.0-py3.10-linux-x86_64.egg`) |
| FlashAttention 4 | 4.0.0b8 |
| Ax / BoTorch | 1.2.1 / 0.16.1 |
| uv | 0.11.6 |

### torchvision Version Trap

**torchvision must match the exact PyTorch CUDA build.** If you see `RuntimeError: operator torchvision::nms does not exist`, it's because torchvision was compiled for a different torch version. For PyTorch 2.8.0+cu128, install `torchvision==0.23.0+cu128`:
```bash
uv pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```
Do NOT let uv auto-resolve — it will pull in a newer torchvision that upgrades torch behind your back. Specify both versions at once.
