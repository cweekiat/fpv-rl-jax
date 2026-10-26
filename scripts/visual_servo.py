"""Tune (small grid search on separate episodes) and evaluate the visual-servoing baseline.

    uv run python scripts/visual_servo.py
"""
import itertools
import json
import os
import sys

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import fmt, run, summarise  # noqa: E402
from fpv.servo import Gains, ServoPilot  # noqa: E402
from train import load_ckpt  # noqa: E402

# the camera-only students' sensors (3 foveated views), benchmark task settings
base = load_ckpt("ckpt/main/B_s0.pkl")[1]._replace(level_mix=False, max_act_delay=2, max_cam_delay=2)
std = base._replace(max_steps=750, sag_floor=0.0)

# tuning on 128 episodes with a different seed from the evaluation
grid = {"cruise_pitch": (0.3, 0.5, 0.7, 0.9), "k_pitch": (3.0, 6.0), "k_yaw": (1.0, 2.0, 4.0),
        "k_climb": (0.5, 1.5, 3.0), "k_los": (0.0, 0.5)}
best, best_r = None, -1
for vals in itertools.product(*grid.values()):
    g = Gains(**dict(zip(grid, vals)))
    r = summarise(jax.jit(lambda k: run(ServoPilot(std, g), k, 128, 1.0, sim=std)[0])(jax.random.PRNGKey(999)))
    if r["success"] > best_r:
        best, best_r = g, r["success"]
        print("new best", round(best_r, 3), dict(zip(grid, vals)), flush=True)

out = {"gains": best._asdict()}
for name, over in {"standard_15s": dict(max_steps=750, sag_floor=0.0), "long_30s": dict(max_steps=1500, sag_floor=0.7)}.items():
    sim = base._replace(**over)
    o = jax.device_get(jax.jit(lambda k: run(ServoPilot(sim, best), k, 1024, 1.0, sim=sim)[0])(jax.random.PRNGKey(321)))
    r = summarise(o)
    m = (o["d0"] >= 180) & (o["d0"] < 230)
    r["success_180_230"] = float(o["success"][m].mean())
    out[name] = r
    print(fmt(f"visual servo {name}", r), flush=True)
with open("results/servo.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results/servo.json")
