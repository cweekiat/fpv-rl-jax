"""Evaluate long-range students on two benchmarks, each pilot with its own sensors:
  * standard: 15 s episodes, plain battery (the paper's long-range task)
  * long:     30 s episodes, battery floored at 70%
Same seeds for every pilot, so drones, targets and noise match.

    uv run python scripts/eval_long.py out_name ckpt/a.pkl ckpt/b.pkl ... [--episodes=1024]
"""
import json
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import fmt, make_pilot, run, summarise  # noqa: E402
from train import load_ckpt  # noqa: E402

BENCHMARKS = {"standard_15s": dict(max_steps=750, sag_floor=0.0), "long_30s": dict(max_steps=1500, sag_floor=0.7)}
BINS = ((100, 140), (140, 180), (180, 230))

args = [a for a in sys.argv[1:] if not a.startswith("--episodes=")]
EPISODES = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--episodes=")), 512)
name, paths = args[0], args[1:]
out = {}
for path in paths:
    pilot = make_pilot(path)
    # the pilot's own sensors (views, tiles), but the benchmark's task settings
    base = load_ckpt(path)[1]._replace(level_mix=False, max_act_delay=2, max_cam_delay=2)
    out[path] = {}
    for bname, over in BENCHMARKS.items():
        sim = base._replace(**over)
        o = jax.device_get(jax.jit(lambda k: run(pilot, k, EPISODES, 1.0, sim=sim)[0])(jax.random.PRNGKey(321)))
        r = summarise(o)
        for lo, hi in BINS:
            m = (o["d0"] >= lo) & (o["d0"] < hi)
            r[f"success_{lo}_{hi}"] = float(o["success"][m].mean())
        to = o["timeout"]
        if to.any():
            r["timeout_never_within_20m"] = float((o["min_dist"][to] > 20).mean())
        out[path][bname] = r
        print(fmt(f"{os.path.basename(path)[:16]} {bname}", r), " far:", f"{r['success_180_230']:.1%}", flush=True)
with open(f"results/{name}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"saved results/{name}.json")
