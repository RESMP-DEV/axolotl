"""Flex attention monkey patch"""

import sys

import torch
import transformers
from packaging import version

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_flex_large_head_dim():
    """Patch Triton autotuning configs for models with head_dim > 256 (e.g. Gemma4's 512).

    The default Inductor configs for head_dim > 256 use num_stages=3 which exceeds
    H100 shared memory (232KB). We reduce num_stages to fit within the hardware limit.
    """
    try:
        from torch._inductor.template_heuristics import (
            CUDAConfigHeuristic,
            FlexConfig,
        )
    except ImportError:
        LOG.warning(
            "Could not import CUDAConfigHeuristic; skipping large head_dim patch")
        return

    _orig_fwd = CUDAConfigHeuristic.get_flex_attn_fwd_configs
    _orig_bwd = CUDAConfigHeuristic.get_flex_attn_bwd_configs

    def _patched_fwd(self, head_dim: int, dtype):
        if head_dim > 256:
            configs = []
            if dtype == torch.float32:
                configs.append(FlexConfig(32, 16, 1, 4))
            else:
                configs.append(FlexConfig(64, 32, 1, 4))
                configs.append(FlexConfig(32, 32, 2, 4))
            return configs
        return _orig_fwd(self, head_dim, dtype)

    def _patched_bwd(self, head_dim: int, dtype):
        if head_dim > 256:
            return [FlexConfig(16, 16, 1, 4)]
        return _orig_bwd(self, head_dim, dtype)

    CUDAConfigHeuristic.get_flex_attn_fwd_configs = _patched_fwd
    CUDAConfigHeuristic.get_flex_attn_bwd_configs = _patched_bwd
    LOG.info("Patched flex attention Triton configs for head_dim > 256")


def patch_flex_wrapper(**flex_attn_compile_kwargs):
    # TODO remove this patch when transformers#37285 is merged and in a release
    is_torch_2_6 = torch.__version__.startswith("2.6")

    if not is_torch_2_6:
        return

    from transformers.utils.import_utils import _torch_version, is_torch_less_or_equal

    from torch.nn.attention.flex_attention import flex_attention

    class WrappedFlexAttention:
        """
        We are doing a singleton class so that flex attention is compiled once when it's first called.
        """

        _instance = None
        _is_flex_compiled = False
        _compiled_flex_attention = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                # Create a new instance if one doesn't already exist
                cls._instance = super().__new__(cls)
            return cls._instance

        @classmethod
        def del_singleton(cls):
            cls._instance = None

        @torch.compiler.disable(recursive=False)
        def __init__(self, training):
            """
            Initialize or update the singleton instance.
            """
            self.training = None
            if not self._is_flex_compiled or training != self.training:
                self.training = training
                if is_torch_less_or_equal("2.5.1"):
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False
                    )
                # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
                # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
                # see https://github.com/pytorch/pytorch/issues/146260 for training
                elif version.parse(_torch_version).base_version == "2.6.0" and training:
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                    )
                # Fallback, usually the most recent torch 2.7.x+ versions
                else:
                    LOG.info(
                        "Compiling flex attention with kwargs: %s. This may take a while...",
                        flex_attn_compile_kwargs,
                    )
                    self._compiled_flex_attention = torch.compile(
                        flex_attention,
                        **flex_attn_compile_kwargs,
                    )
                    LOG.info("Flex attention compiled successfully.")

                self._is_flex_compiled = True

        def __call__(self):
            return self._compiled_flex_attention

    transformers.integrations.flex_attention.WrappedFlexAttention = WrappedFlexAttention
    sys.modules[
        "transformers.integrations.flex_attention"
    ].WrappedFlexAttention = WrappedFlexAttention
