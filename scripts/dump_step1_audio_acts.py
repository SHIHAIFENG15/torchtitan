#!/usr/bin/env python3
"""Dump one step of Muse audio-encoder intermediates.

Same AllGatherKV debug config and the same step-0 checkpoint as the
fp32-attention run. The model code is unchanged. This records the audio
tower stage by stage so the first divergent op can be read off the files.
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


_SAVED: dict[str, list[torch.Tensor]] = {}
_META: dict[str, object] = {}
_OPEN = False
_DONE = False
_STAMPED = False


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu()


def _append(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor):
        return
    import spmd_types as spmd

    with spmd.local():
        _SAVED.setdefault(name, []).append(_to_cpu(tensor))


def _stamp(engine) -> None:
    global _STAMPED
    if _STAMPED:
        return
    from torchtitan.models.base.module import ModuleDict
    from torchtitan.models.muse_spark.audio_encoder import MuseSparkAudioEncoder
    from torchtitan.models.muse_spark.model import MuseSparkTransformerBlock

    found_audio = False
    found_layer = False
    for part_index, part in enumerate(engine.model_parts):
        prefix = "" if len(engine.model_parts) == 1 else f"part{part_index}."
        for name, module in part.named_modules():
            full = prefix + name
            if isinstance(module, MuseSparkAudioEncoder) or (
                "audio_encoder" in name and not isinstance(module, ModuleDict)
            ):
                _wrap(module, full)
                found_audio = True
            if isinstance(module, MuseSparkTransformerBlock) and name.endswith("layers.0"):
                _wrap_block(module)
                found_layer = True
    if not found_audio or not found_layer:
        raise RuntimeError("audio encoder or layers.0 was not found")
    _STAMPED = True


def _wrap(module, name: str) -> None:
    if getattr(module, "_audio_dump_wrapped", False):
        return
    module._audio_dump_wrapped = True
    original = module.forward
    from torchtitan.models.muse_spark.audio_encoder import MuseSparkAudioEncoder

    def wrapped(*args, **kwargs):
        if _OPEN and isinstance(module, MuseSparkAudioEncoder) and args:
            _append(name + ".waveforms", args[0])
            lengths = kwargs.get("waveform_padded_lengths_N")
            if torch.is_tensor(lengths):
                _append(name + ".lengths", lengths)
        out = original(*args, **kwargs)
        if _OPEN and torch.is_tensor(out):
            _append(name, out)
        return out

    module.forward = wrapped


def _wrap_block(module) -> None:
    if getattr(module, "_audio_dump_block_wrapped", False):
        return
    module._audio_dump_block_wrapped = True
    original = module.forward

    def wrapped(*args, **kwargs):
        if _OPEN and args and "block_input" not in _SAVED:
            block_input = args[0]
            _append("block_input", block_input.reshape(-1, block_input.shape[-1]))
        return original(*args, **kwargs)

    module.forward = wrapped


def _install() -> None:
    from torchtitan.models.muse_spark.model import MuseSparkModel
    from torchtitan.training_engine import TrainingEngine

    original_model = MuseSparkModel.forward

    def model_forward(self, tokens, positions=None, *args, **kwargs):
        global _OPEN, _DONE
        record = not _DONE and bool(os.environ.get("STEP1_AUDIO_DIR"))
        if record:
            _OPEN = True
            _META["audio_is_dummy"] = kwargs.get("waveforms") is None
            indices = kwargs.get("audio_bank_indices_T")
            if torch.is_tensor(indices):
                _append("audio_bank_indices_T", indices.reshape(-1))
        try:
            return original_model(self, tokens, positions, *args, **kwargs)
        finally:
            if record:
                _OPEN = False
                _DONE = True

    MuseSparkModel.forward = model_forward

    original_bank = MuseSparkModel._get_audio_bank

    def get_audio_bank(self, *, waveforms_SC, waveform_padded_lengths_N):
        out = original_bank(
            self,
            waveforms_SC=waveforms_SC,
            waveform_padded_lengths_N=waveform_padded_lengths_N,
        )
        if _OPEN:
            _append("audio_embeds", out)
        return out

    MuseSparkModel._get_audio_bank = get_audio_bank

    original_forward_backward = TrainingEngine.forward_backward

    def forward_backward(self, *args, **kwargs):
        _stamp(self)
        return original_forward_backward(self, *args, **kwargs)

    TrainingEngine.forward_backward = forward_backward

    original_optim = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0:
            if "audio_embeds" not in _SAVED or "block_input" not in _SAVED:
                raise RuntimeError("step1 audio dump missed")
            _write()
        return original_optim(self)

    TrainingEngine.optim_step = optim_step


def _write() -> None:
    out_dir = os.environ["STEP1_AUDIO_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    payload = {
        "rank": rank,
        "world_size": world,
        "config": os.environ.get("STEP1_AUDIO_CONFIG", ""),
        "git_rev": os.environ.get("STEP1_AUDIO_GIT", ""),
        "torch_version": torch.__version__,
        "audio_is_dummy": bool(_META.get("audio_is_dummy", False)),
        "tensors": _SAVED,
    }
    stem = os.path.join(out_dir, f"rank{rank}")
    torch.save(payload, stem + ".pt.tmp")
    os.replace(stem + ".pt.tmp", stem + ".pt")
    embeds = _SAVED["audio_embeds"][0]
    print(
        f"STEP1_AUDIO rank={rank} world={world} dummy={payload['audio_is_dummy']} "
        f"keys={len(_SAVED)} audio_embeds={tuple(embeds.shape)} {embeds.dtype} "
        f"out={stem}.pt",
        flush=True,
    )


def _encoder_key(names: set[str]) -> str | None:
    for name in names:
        if name == "audio_encoder" or name.endswith(".audio_encoder"):
            return name
    return None


def _check(out_dir: str) -> int:
    for rank in range(8):
        path = os.path.join(out_dir, f"rank{rank}.pt")
        if not os.path.isfile(path):
            print(f"missing {path}", file=sys.stderr)
            return 3
        payload = torch.load(path, weights_only=False)
        tensors = payload["tensors"]
        names = set(tensors)
        encoder = _encoder_key(names)
        convs = sorted(name for name in names if name.endswith(".conv"))
        blocks = sorted(
            name for name in names if name.endswith(".block1") or name.endswith(".block2")
        )
        missing = [
            label
            for label, ok in (
                ("audio_embeds", "audio_embeds" in names),
                ("block_input", "block_input" in names),
                ("audio_encoder", encoder is not None),
                ("conv", bool(convs)),
                ("block", bool(blocks)),
            )
            if not ok
        ]
        if missing:
            print(f"rank{rank} missing {missing}", file=sys.stderr)
            return 3
        for name in ("audio_embeds", "block_input", encoder):
            if not tensors[name]:
                print(f"rank{rank} empty {name}", file=sys.stderr)
                return 3
        embeds = tensors["audio_embeds"][0]
        block_input = tensors["block_input"][0]
        print(
            f"rank{rank} dummy={payload['audio_is_dummy']} "
            f"audio_embeds={tuple(embeds.shape)} {embeds.dtype} "
            f"block_input={tuple(block_input.shape)} {block_input.dtype} "
            f"convs={len(convs)} blocks={len(blocks)} calls={len(tensors[encoder])}",
            flush=True,
        )
    print("STEP1_AUDIO check ok", flush=True)
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

    out_dir = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out_dir, "rank0.pt")):
        print(f"refusing to overwrite {out_dir}", file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)
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
    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env["STEP1_AUDIO_WORKER"] = "1"
    env["STEP1_AUDIO_DIR"] = out_dir
    env["STEP1_AUDIO_CONFIG"] = args.config
    env["STEP1_AUDIO_GIT"] = git_rev
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
        f"audio acts dump module={args.module} config={args.config} "
        f"ngpu={args.ngpu} load={load_path} out={out_dir}",
        flush=True,
    )
    code = subprocess.run(cmd, cwd=root, env=env).returncode
    if code != 0:
        return code
    return _check(out_dir)


if __name__ == "__main__":
    if os.environ.get("STEP1_AUDIO_WORKER") == "1":
        _worker()
    else:
        raise SystemExit(_parent())
