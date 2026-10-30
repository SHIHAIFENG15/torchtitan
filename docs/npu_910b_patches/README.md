# Ascend 910B flex / remat patches (reusable)

These patches are **not** TorchTitan source changes. They patch installed
site-packages so Muse Spark debug recipes can finish on **Ascend 910B4**
(non-950). Ascend 950 keeps the stock mask-out / decoding paths.

## Target versions

| Package | Version |
| --- | --- |
| `torch` | `2.15.0+cpu` (PrivateUse1 / NPU via torch_npu) |
| `torch_npu` | `2.15.0.dev20261005+git14dc54e` |
| `torch_remat` | `0.2.2` |

Upstream baselines were taken from that `torch_npu` wheel and the
`torch_remat==0.2.2` wheel. Expected MD5 after apply:

| File under site-packages | MD5 |
| --- | --- |
| `torch_npu/_inductor/kernel/flex_attention.py` | `659bf1dbc6c9652d8a5fdf9dad9bedeb` |
| `torch_npu/_inductor/runtime/triton_heuristics.py` | `e0fcd810ba31c602be44fb7c5e49b058` |
| `torch_remat/_placeholder.py` | `f3d63fe46a039b58da3c39bdd3152e6a` |

## What each patch does

1. `flex_attention.py.patch`
   - Non-950: disable mask-out; disable flex decoding; default backward tiles to 16.
2. `triton_heuristics.py.patch`
   - Non-950: do not emit `simt_only` for `simd_simt_mix`.
   - Serialize multi-rank `flex_attention` autotune with
     `/tmp/npu_flex_attention_autotune.lock`.
3. `torch_remat_placeholder.py.patch`
   - On NPU, build remat placeholders with `empty_strided` instead of a null
     storage pointer (avoids allocator SIGSEGV under TP remat).

## Apply

From a checkout that has these files, with the training venv active:

```bash
# optional: point at another prefix
# export NPU_PATCH_SITE_PACKAGES=/path/to/site-packages
bash docs/npu_910b_patches/apply_npu_910b_patches.sh
```

The script backs up each target as `*.pre_npu_910b_patch`, applies with
`patch -p1`, then checks MD5.

Also source the compile-thread limits before training (or add them to your
container `env.sh`):

```bash
source docs/npu_910b_patches/env_npu_910b.sh
```

## TorchTitan commits still required

Apply these patches **and** use the NPU Titan branch (for example
`npu-muse-spark-debug-r04`): RoPE complex64 indexing, SwiGLU fallback,
vision `contiguous()`, loss compile-time assert skip, recipe timeouts, etc.
See `docs/npu_pta_fixes.md`.

## Dry-run / restore

```bash
bash docs/npu_910b_patches/apply_npu_910b_patches.sh --dry-run
bash docs/npu_910b_patches/apply_npu_910b_patches.sh --restore
```
