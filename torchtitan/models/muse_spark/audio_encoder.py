# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.base.module import Module, ModuleDict

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm


def _pairwise_mean(values: torch.Tensor) -> torch.Tensor:
    """Mean over the last dimension with a fixed pairwise add tree.

    Audio hidden sizes are powers of two. CUDA and NPU fused rms_norm
    kernels use different trees, so a value within 1e-7 of a bf16 midpoint
    can round to opposite bins.
    """
    width = values.shape[-1]
    if width & (width - 1):
        raise ValueError(f"audio RMSNorm width must be a power of two, got {width}")
    reduced = values
    while reduced.shape[-1] > 1:
        reduced = reduced.reshape(*reduced.shape[:-1], reduced.shape[-1] // 2, 2).sum(-1)
    return reduced / width


class AudioRMSNorm(RMSNorm):
    """fp32 RMSNorm, then one rounding back to the input dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        values = input.float()
        normalized = values * torch.rsqrt(_pairwise_mean(values * values) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(input.dtype)


# Shape suffix legend:
# N = waveforms, S = packed waveform samples, L = sequence length,
# C = audio or stage input channels, P = patches,
# F = flattened patch or MLP features, T = packed output tokens,
# D = hidden dimension, O = output dimension.


class _AudioPatchifier(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        patch_size: int
        linear: Linear.Config
        norm: AudioRMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.linear = config.linear.build()
        self.norm = config.norm.build()

    def forward(self, value_LC: torch.Tensor) -> torch.Tensor:
        length, channels = value_LC.shape
        patches_PF = value_LC.reshape(
            length // self.patch_size,
            self.patch_size * channels,
        )
        value_PD = self.linear(patches_PF)
        remat.recompute_needs_tensor(value_PD)
        return self.norm(value_PD)


class _DepthwiseCausalConv(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kernel_size: int
        conv: Conv1d.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.kernel_size = config.kernel_size
        self.conv = config.conv.build()

    def forward(
        self,
        value_LD: torch.Tensor,
    ) -> torch.Tensor:
        conv_input_DL = F.pad(
            value_LD.transpose(0, 1),
            (self.kernel_size - 1, 0),
        )
        return self.conv(conv_input_DL).transpose(0, 1)


class _AudioBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        conv: _DepthwiseCausalConv.Config
        norm: AudioRMSNorm.Config
        linear1: Linear.Config
        linear2: Linear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv = config.conv.build()
        self.norm = config.norm.build()
        self.linear1 = config.linear1.build()
        self.linear2 = config.linear2.build()

    def forward(self, value_LD: torch.Tensor) -> torch.Tensor:
        residual_LD = value_LD
        value_LD = self.conv(value_LD)
        hidden_LF = self.linear1(self.norm(value_LD))
        remat.recompute_needs_tensor(hidden_LF)
        value_LD = self.linear2(F.gelu(hidden_LF))
        remat.recompute_needs_tensor(value_LD)
        return value_LD + residual_LD


class _AudioStage(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        patchifier: _AudioPatchifier.Config
        block1: _AudioBlock.Config
        block2: _AudioBlock.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.patchifier: _AudioPatchifier = config.patchifier.build()
        self.block1: _AudioBlock = config.block1.build()
        self.block2: _AudioBlock = config.block2.build()

    def forward(self, value_LC: torch.Tensor) -> torch.Tensor:
        value_PD = self.patchifier(value_LC)
        value_PD = self.block1(value_PD)
        return self.block2(value_PD)


class MuseSparkAudioEncoder(Module):
    """Encode packed waveforms into one packed sequence of audio features."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        output_size: int
        patch_size: int
        input_channels: int
        layers: list[_AudioStage.Config]
        norm: AudioRMSNorm.Config
        o_proj: Linear.Config
        dual_norm: AudioRMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()
        self.norm = config.norm.build()
        self.o_proj = config.o_proj.build()
        self.dual_norm = config.dual_norm.build()

    def _encode(self, value_LC: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            value_LC = layer(value_LC)
        return self.dual_norm(self.o_proj(self.norm(value_LC)))

    def forward(
        self,
        waveforms_SC: torch.Tensor,
        *,
        waveform_padded_lengths_N: torch.Tensor,
    ) -> torch.Tensor:
        """Encode patch-aligned waveform segments and concatenate their outputs.

        ``waveform_padded_lengths_N`` gives each segment's padded sample count.
        Segments are encoded independently so causal convolutions cannot cross
        waveform boundaries. The returned tensor contains all output tokens in
        waveform order with shape ``(sum(length / patch_size), output_size)``.
        """
        waveforms = waveforms_SC.split(waveform_padded_lengths_N.tolist())

        # TODO: If we need to squeeze performance on audio encoder,
        # process packed waveforms in one encoder pass using
        # attn_gym.linear.causal_conv1d with per-waveform cu_seqlens.
        # The Qwen3.5 wrapper cannot be reused directly because it hardcodes
        # activation="silu" and omits bias.
        return torch.cat([self._encode(value_LC) for value_LC in waveforms])
