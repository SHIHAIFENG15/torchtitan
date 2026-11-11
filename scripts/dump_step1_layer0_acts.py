#!/usr/bin/env python3
"""Dump the layer-0 activations that produce the Muse router gate input.

One step, one config. The first forward is unchanged. Hooks only record
tensors from layers.0, then ffn_input is checked against the gate dump.
"""

import argparse
import os
import subprocess
import sys

import torch
from torch.distributed.tensor import DTensor
import torch.distributed as dist


_LAYER: dict[str, torch.Tensor] = {}
_MODEL: dict[str, torch.Tensor] = {}
_SEEN = False
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
    from torchtitan.models.muse_spark.model import MuseSparkTransformerBlock

    found = False
    for part_index, part in enumerate(engine.model_parts):
        for name, module in part.named_modules():
            if isinstance(module, MuseSparkTransformerBlock) and name.endswith("layers.0"):
                prefix = "" if len(engine.model_parts) == 1 else f"part{part_index}."
                module._layer0_dump_name = prefix + name
                found = True
    if not found:
        raise RuntimeError("layers.0 MuseSparkTransformerBlock not found")
    _STAMPED = True


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu()


def _flatten_tokens(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def _install() -> None:
    from torchtitan.models.muse_spark.model import MuseSparkModel, MuseSparkTransformerBlock
    from torchtitan.training_engine import TrainingEngine

    original_model = MuseSparkModel.forward

    def model_forward(self, tokens, positions=None, *args, **kwargs):
        if "tokens" not in _MODEL and os.environ.get("STEP1_LAYER0_DUMP_DIR"):
            _MODEL["tokens"] = tokens.detach()
            if positions is not None:
                _MODEL["positions"] = positions.detach()
            for key in ("image_bank_indices_T", "video_bank_indices_T"):
                value = kwargs.get(key)
                if torch.is_tensor(value):
                    _MODEL[key] = value.detach()
        return original_model(self, tokens, positions, *args, **kwargs)

    MuseSparkModel.forward = model_forward

    original_forward = MuseSparkTransformerBlock.forward

    def forward(
        self,
        x,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        global _SEEN
        record = (
            bool(getattr(self, "_layer0_dump_name", None))
            and not _SEEN
            and bool(os.environ.get("STEP1_LAYER0_DUMP_DIR"))
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
            return original_forward(
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
                _SEEN = True
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
                    raise RuntimeError("layer0 dump missed " + ",".join(missing))
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

    MuseSparkTransformerBlock.forward = forward

    original_forward_backward = TrainingEngine.forward_backward

    def forward_backward(self, *args, **kwargs):
        _stamp(self)
        return original_forward_backward(self, *args, **kwargs)

    TrainingEngine.forward_backward = forward_backward

    original_optim = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0 and _LAYER:
            _write()
        return original_optim(self)

    TrainingEngine.optim_step = optim_step


def _write() -> None:
    out_dir = os.environ["STEP1_LAYER0_DUMP_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    payload = {
        "rank": rank,
        "world_size": world,
        "config": os.environ.get("STEP1_LAYER0_DUMP_CONFIG", ""),
        "git_rev": os.environ.get("STEP1_LAYER0_DUMP_GIT", ""),
        "torch_version": torch.__version__,
        "layer0": _LAYER,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    print(
        f"STEP1_LAYER0_DUMP rank={rank} world={world} keys={len(_LAYER)} out={stem}.pt",
        flush=True,
    )


def _check(out_dir: str, expect_dir: str) -> int:
    required = (
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
    mismatched = 0
    compared = 0
    for rank in range(8):
        path = os.path.join(out_dir, f"rank{rank}.pt")
        if not os.path.isfile(path):
            print(f"missing {path}", file=sys.stderr)
            return 3
        entry = torch.load(path, weights_only=False)["layer0"]
        for key in required:
            if key not in entry:
                print(f"rank{rank} missing {key}", file=sys.stderr)
                return 3
        ffn = entry["ffn_input"]
        tokens = ffn.shape[0]
        if entry["padding_mask"].shape != (tokens,):
            print(f"rank{rank} padding_mask {tuple(entry['padding_mask'].shape)}", file=sys.stderr)
            return 3
        for key in _STREAMS:
            if entry[key].shape[0] != tokens:
                print(f"rank{rank} {key} {tuple(entry[key].shape)}", file=sys.stderr)
                return 3
        if entry["query"].shape[0] != tokens or entry["key"].shape[0] != tokens:
            print(
                f"rank{rank} query {tuple(entry['query'].shape)} key {tuple(entry['key'].shape)}",
                file=sys.stderr,
            )
            return 3
        if entry["block_input"].dtype != torch.float32 or entry["stream"].dtype != torch.float32:
            print(f"rank{rank} stream dtype {entry['block_input'].dtype}", file=sys.stderr)
            return 3
        if ffn.dtype != torch.bfloat16:
            print(f"rank{rank} ffn_input dtype {ffn.dtype}", file=sys.stderr)
            return 3
        expect_path = os.path.join(expect_dir, f"rank{rank}.pt")
        if not os.path.isfile(expect_path):
            print(f"missing gate baseline {expect_path}", file=sys.stderr)
            return 3
        expect = torch.load(expect_path, weights_only=False)["routers"]
        gate = expect["layers.0.feed_forward.router"]["gate_input"]
        compared += gate.numel()
        if gate.shape != ffn.shape or gate.dtype != ffn.dtype or not torch.equal(gate, ffn):
            mismatched += gate.numel()
            if gate.shape == ffn.shape:
                delta = (gate.float() - ffn.float()).abs().max()
                print(f"rank{rank} ffn_input max abs {float(delta):.4e}", flush=True)
            else:
                print(f"rank{rank} ffn_input {tuple(ffn.shape)} gate {tuple(gate.shape)}", flush=True)
        print(
            f"rank{rank} tokens={tokens} ffn={tuple(ffn.shape)} query={tuple(entry['query'].shape)}",
            flush=True,
        )
    if compared == 0:
        print(f"no gate baseline in {expect_dir}", file=sys.stderr)
        return 3
    print(f"ffn_input mismatch {mismatched}/{compared} vs {expect_dir}", flush=True)
    if mismatched:
        return 3
    print("STEP1_LAYER0_DUMP check ok", flush=True)
    return 0


def _worker() -> None:
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
    parser.add_argument(
        "--expect-gate",
        default="outputs/step1_gate_scores/mm_fsdp2_tp2_cp2_ep4",
    )
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
    env["STEP1_LAYER0_DUMP_WORKER"] = "1"
    env["STEP1_LAYER0_DUMP_DIR"] = out_dir
    env["STEP1_LAYER0_DUMP_CONFIG"] = args.config
    env["STEP1_LAYER0_DUMP_GIT"] = git_rev
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
        f"layer0 dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    code = subprocess.run(cmd, cwd=root, env=env).returncode
    if code != 0:
        return code
    return _check(out_dir, os.path.abspath(args.expect_gate))


if __name__ == "__main__":
    if os.environ.get("STEP1_LAYER0_DUMP_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
