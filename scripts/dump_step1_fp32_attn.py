#!/usr/bin/env python3
"""Dump one step of the fp32-attention Muse run.

Same AllGatherKV debug config and the same step-0 checkpoint. The model
change is the thing under test, so this does not compare against the older
bf16-qkv expert ids or loss. It records layer-0 activations and every
router's expert ids from that single step.
"""

import argparse
import os
import subprocess
import sys

import torch
from torch.distributed.tensor import DTensor
import torch.distributed as dist


_LAYER: dict[str, torch.Tensor] = {}
_ROUTERS: dict[str, dict[str, torch.Tensor]] = {}
_MODEL: dict[str, torch.Tensor] = {}
_LAYER_SEEN = False
_ROUTER_SEEN: set[str] = set()
_STAMPED = False

_STREAMS = (
    "block_input",
    "attention_input",
    "attention_output",
    "attention_branch",
    "stream",
    "ffn_input",
)


def _stamp(engine) -> None:
    global _STAMPED
    if _STAMPED:
        return
    from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter
    from torchtitan.models.muse_spark.model import MuseSparkTransformerBlock

    found_layer = False
    for part_index, part in enumerate(engine.model_parts):
        prefix = "" if len(engine.model_parts) == 1 else f"part{part_index}."
        for name, module in part.named_modules():
            if isinstance(module, MuseSparkTransformerBlock) and name.endswith("layers.0"):
                module._fp32_attn_layer_name = prefix + name
                found_layer = True
            if isinstance(module, TokenChoiceTopKRouter):
                module._fp32_attn_router_name = prefix + name
    if not found_layer:
        raise RuntimeError("layers.0 MuseSparkTransformerBlock not found")
    _STAMPED = True


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu()


def _flatten_tokens(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def _install() -> None:
    from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter
    from torchtitan.models.muse_spark.model import MuseSparkModel, MuseSparkTransformerBlock
    from torchtitan.training_engine import TrainingEngine

    original_model = MuseSparkModel.forward

    def model_forward(self, tokens, positions=None, *args, **kwargs):
        if "tokens" not in _MODEL and os.environ.get("STEP1_FP32_ATTN_DIR"):
            _MODEL["tokens"] = tokens.detach()
            if positions is not None:
                _MODEL["positions"] = positions.detach()
            for key in (
                "image_bank_indices_T",
                "video_bank_indices_T",
                "audio_bank_indices_T",
            ):
                value = kwargs.get(key)
                if torch.is_tensor(value):
                    _MODEL[key] = value.detach()
        return original_model(self, tokens, positions, *args, **kwargs)

    MuseSparkModel.forward = model_forward

    original_block = MuseSparkTransformerBlock.forward

    def block_forward(
        self,
        x,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        global _LAYER_SEEN
        record = (
            bool(getattr(self, "_fp32_attn_layer_name", None))
            and not _LAYER_SEEN
            and bool(os.environ.get("STEP1_FP32_ATTN_DIR"))
        )
        captured: dict[str, torch.Tensor] = {}
        restores: list[tuple[object, str, object]] = []

        def patch(obj, attr, fn) -> None:
            restores.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, fn)

        if record:
            captured["block_input"] = x.detach()
            if positions is not None:
                captured["positions"] = positions.detach()
            if padding_mask is not None:
                captured["padding_mask"] = padding_mask.detach()
            attention = self.attention
            orig_attn = attention.forward
            orig_qkv = attention.qkv_linear.forward
            orig_prepare = attention._prepare_query_and_key
            orig_inner = attention.inner_attention.forward
            orig_branch = self.post_attention_norm.forward
            orig_residual = self.attention_residual.forward
            orig_ffn = self.feed_forward.forward

            def attention_forward(*args, **kwargs):
                if args and "attention_input" not in captured:
                    captured["attention_input"] = args[0].detach()
                out = orig_attn(*args, **kwargs)
                captured.setdefault("attention_output", out.detach())
                return out

            def qkv_forward(*args, **kwargs):
                out = orig_qkv(*args, **kwargs)
                captured.setdefault("value", out[2].detach())
                return out

            def prepare(*args, **kwargs):
                query, key = orig_prepare(*args, **kwargs)
                captured.setdefault("query", query.detach())
                captured.setdefault("key", key.detach())
                return query, key

            def inner_forward(*args, **kwargs):
                out = orig_inner(*args, **kwargs)
                captured.setdefault("attn_out", out.detach())
                return out

            def branch_forward(*args, **kwargs):
                out = orig_branch(*args, **kwargs)
                captured.setdefault("attention_branch", out.detach())
                return out

            def residual_forward(*args, **kwargs):
                out = orig_residual(*args, **kwargs)
                captured.setdefault("stream", out.detach())
                return out

            def ffn_forward(*args, **kwargs):
                if args and "ffn_input" not in captured:
                    captured["ffn_input"] = args[0].detach()
                return orig_ffn(*args, **kwargs)

            patch(attention, "forward", attention_forward)
            patch(attention.qkv_linear, "forward", qkv_forward)
            patch(attention, "_prepare_query_and_key", prepare)
            patch(attention.inner_attention, "forward", inner_forward)
            patch(self.post_attention_norm, "forward", branch_forward)
            patch(self.attention_residual, "forward", residual_forward)
            patch(self.feed_forward, "forward", ffn_forward)
        try:
            return original_block(
                self,
                x,
                attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        finally:
            for obj, attr, old in reversed(restores):
                setattr(obj, attr, old)
            if record:
                _LAYER_SEEN = True
                import spmd_types as spmd

                required = (
                    "block_input",
                    "attention_input",
                    "query",
                    "key",
                    "value",
                    "attn_out",
                    "attention_output",
                    "attention_branch",
                    "stream",
                    "ffn_input",
                    "padding_mask",
                    "positions",
                )
                missing = [key for key in required if key not in captured]
                if "tokens" not in _MODEL:
                    missing.append("tokens")
                if missing:
                    raise RuntimeError("fp32 attn dump missed " + ",".join(missing))
                with spmd.local():
                    entry = {}
                    for key, value in captured.items():
                        entry[key] = _to_cpu(value)
                    for key, value in _MODEL.items():
                        if key not in entry:
                            entry[key] = _to_cpu(value)
                    for key in _STREAMS:
                        entry[key] = _flatten_tokens(entry[key])
                    entry["padding_mask"] = entry["padding_mask"].reshape(-1).to(torch.bool)
                    entry["tokens"] = entry["tokens"].reshape(-1)
                _LAYER.update(entry)

    MuseSparkTransformerBlock.forward = block_forward

    original_router = TokenChoiceTopKRouter.forward

    def router_forward(self, x_TD, expert_bias_E=None, *, padding_mask_T=None, **kwargs):
        name = getattr(self, "_fp32_attn_router_name", None)
        record = bool(name) and name not in _ROUTER_SEEN and bool(os.environ.get("STEP1_FP32_ATTN_DIR"))
        result = original_router(
            self,
            x_TD,
            expert_bias_E,
            padding_mask_T=padding_mask_T,
            **kwargs,
        )
        if record:
            _ROUTER_SEEN.add(name)
            import spmd_types as spmd

            with spmd.local():
                topk_ids = result[1].detach()
                entry = {
                    "expert_ids": torch.sort(_to_cpu(topk_ids).to(torch.int32), dim=-1).values,
                    "topk_scores": _to_cpu(result[0]).to(torch.float32),
                }
                if padding_mask_T is not None:
                    entry["padding_mask"] = _to_cpu(padding_mask_T).reshape(-1).to(torch.bool)
            _ROUTERS[name] = entry
        return result

    TokenChoiceTopKRouter.forward = router_forward

    original_forward_backward = TrainingEngine.forward_backward

    def forward_backward(self, *args, **kwargs):
        _stamp(self)
        return original_forward_backward(self, *args, **kwargs)

    TrainingEngine.forward_backward = forward_backward

    original_optim = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0 and _LAYER and _ROUTERS:
            _write()
        return original_optim(self)

    TrainingEngine.optim_step = optim_step


def _write() -> None:
    out_dir = os.environ["STEP1_FP32_ATTN_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    payload = {
        "rank": rank,
        "world_size": world,
        "config": os.environ.get("STEP1_FP32_ATTN_CONFIG", ""),
        "git_rev": os.environ.get("STEP1_FP32_ATTN_GIT", ""),
        "torch_version": torch.__version__,
        "layer0": _LAYER,
        "routers": _ROUTERS,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    query = _LAYER.get("query")
    value = _LAYER.get("value")
    attn = _LAYER.get("attn_out")
    print(
        f"STEP1_FP32_ATTN rank={rank} world={world} routers={len(_ROUTERS)} "
        f"query_dtype={None if query is None else query.dtype} "
        f"value_dtype={None if value is None else value.dtype} "
        f"attn_dtype={None if attn is None else attn.dtype} out={stem}.pt",
        flush=True,
    )


def _check(out_dir: str) -> int:
    layer_keys = (
        "tokens",
        "positions",
        "padding_mask",
        "block_input",
        "attention_input",
        "query",
        "key",
        "value",
        "attn_out",
        "attention_output",
        "attention_branch",
        "stream",
        "ffn_input",
    )
    router_names = [f"layers.{i}.feed_forward.router" for i in range(8)]
    fp32_keys = ("query", "key", "value", "attn_out", "attention_output")
    failed = False
    for rank in range(8):
        path = os.path.join(out_dir, f"rank{rank}.pt")
        if not os.path.isfile(path):
            print(f"missing {path}", file=sys.stderr)
            return 3
        payload = torch.load(path, weights_only=False)
        layer = payload["layer0"]
        routers = payload["routers"]
        for key in layer_keys:
            if key not in layer:
                print(f"rank{rank} missing layer0.{key}", file=sys.stderr)
                return 3
        dtypes = {key: str(layer[key].dtype).replace("torch.", "") for key in fp32_keys}
        print(
            f"rank{rank} query={tuple(layer['query'].shape)} dtypes={dtypes} "
            f"routers={len(routers)}",
            flush=True,
        )
        for key in fp32_keys:
            if layer[key].dtype != torch.float32:
                print(f"rank{rank} {key} dtype {layer[key].dtype}", file=sys.stderr)
                failed = True
        if set(routers) != set(router_names):
            print(f"rank{rank} routers={sorted(routers)}", file=sys.stderr)
            return 3
        for name in router_names:
            if "expert_ids" not in routers[name]:
                print(f"rank{rank} {name} missing expert_ids", file=sys.stderr)
                return 3
    if failed:
        print("STEP1_FP32_ATTN dtype check failed", flush=True)
        return 3
    print("STEP1_FP32_ATTN check ok", flush=True)
    return 0


def _worker() -> None:
    if os.environ.get("MUSE_SETUP_ASCEND") == "1":
        import torchtitan_ascend

        torchtitan_ascend.setup()
    _install()
    from torchtitan.train import main

    main()


def _parent() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ngpu", required=True, type=int)
    parser.add_argument("--load", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tb", required=True)
    args = parser.parse_args()
    load_path = os.path.abspath(args.load)
    if not os.path.isdir(load_path):
        print(f"missing checkpoint: {load_path}", file=sys.stderr)
        return 2

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from _checkpoint_test_config import configure_training_run

    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (root, prior) if item)
    env["TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH"] = load_path
    module, config = configure_training_run(
        env,
        module=args.module,
        config=args.config,
        steps=1,
        tb_folder=args.tb,
        checkpoint_mode="load",
        export_dtype="bfloat16",
    )
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env["STEP1_FP32_ATTN_WORKER"] = "1"
    env["STEP1_FP32_ATTN_DIR"] = out_dir
    env["STEP1_FP32_ATTN_CONFIG"] = args.config
    env["STEP1_FP32_ATTN_GIT"] = git_rev
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["LOG_RANK"] = env.get("LOG_RANK", "0")
    for key, value in (
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("TORCH_DEVICE_BACKEND_AUTOLOAD", "1"),
        ("MUSE_DEBUG_DETERMINISTIC", "1"),
    ):
        env.setdefault(key, value)
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(args.ngpu),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--local-ranks-filter",
        env["LOG_RANK"],
        "--role",
        "rank",
        "--tee",
        "3",
        os.path.abspath(__file__),
        "--module",
        module,
        "--config",
        config,
    ]
    print(
        f"fp32 attn dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    code = subprocess.run(cmd, cwd=root, env=env).returncode
    if code != 0:
        return code
    return _check(out_dir)


if __name__ == "__main__":
    if os.environ.get("STEP1_FP32_ATTN_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
