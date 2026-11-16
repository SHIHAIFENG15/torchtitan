#!/usr/bin/env python3
"""Load a committed step-0 checkpoint and train the recipe for N steps.

Uses scripts/_checkpoint_test_config.py, the same helper the 1000-step GPU
baselines used. Does not change the recipe module.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from _checkpoint_test_config import configure_training_run

    module = os.environ["ACC_MODULE"]
    config = os.environ["ACC_CONFIG"]
    load_path = os.path.abspath(os.environ["ACC_LOAD"])
    steps = int(os.environ.get("ACC_STEPS", "1000"))
    ngpu = os.environ["ACC_NGPU"]
    if not os.path.isdir(load_path):
        print(f"missing checkpoint: {load_path}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH"] = load_path
    gen_module, gen_config = configure_training_run(
        env,
        module=module,
        config=config,
        steps=steps,
        tb_folder=os.environ.get("ACC_TB", "tb_acc"),
        checkpoint_mode="load",
        export_dtype="bfloat16",
    )
    env["NGPU"] = ngpu
    env["MODULE"] = gen_module
    env["CONFIG"] = gen_config
    print(
        f"acc module={module} config={config} steps={steps} "
        f"ngpu={ngpu} load={load_path}",
        flush=True,
    )
    return subprocess.run(["./run_train.sh"], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
