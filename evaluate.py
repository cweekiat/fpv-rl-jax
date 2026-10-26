"""Evaluation used for the README and the paper.

    uv run python evaluate.py main        # teacher vs student, DR on/off, flights.png, fpv_view.gif
    uv run python evaluate.py sweep       # robustness outside the training ranges (fig6)
    uv run python evaluate.py ablations   # every checkpoint in ckpt/ under full DR
    uv run python evaluate.py long        # long range: teacher vs camera-only student, lr_flights

Results are written to results/*.json and figures to docs/figures/.
"""
import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fpv import env
from fpv.dagger import make_env_fns, student_act
from fpv.ppo import policy
from train import load_ckpt

# Longer latency buffers than in training so the sweep can go past the training range.
EVAL_SIM = env.Sim(max_act_delay=4, max_cam_delay=4)
FIG, RES = "docs/figures", "results"


def make_pilot(path):
    params, sim = load_ckpt(path)
    sim = sim._replace(max_act_delay=EVAL_SIM.max_act_delay, max_cam_delay=EVAL_SIM.max_cam_delay)
    if "actor" in params:
        return lambda s, fr: (policy(params, jax.vmap(lambda x: env.teacher_obs(x, sim))(s))[0], None)
    if sim.recurrent:
        return RecurrentPilot(params, sim)
    return lambda s, fr: student_act(params, fr, s, sim)[:2]


class RecurrentPilot:
    """A student with memory: `act` threads the GRU state, which `run` resets per episode."""

    def __init__(self, params, sim):
        self.params, self.sim = params, sim

    def init(self, n):
        return jnp.zeros((n, self.sim.gru_size))

    def act(self, s, fr, h):
        a, g, _, _, h = student_act(self.params, fr, s, self.sim, h)
        return a, g, h


def run(pilot, key, n, dr_scale=1.0, override=None, log=False, sim=EVAL_SIM, level=1.0):
    """Fly the first episode of n environments to completion."""
    init_envs, env_step = make_env_fns(sim, dr_scale)
    k_init, k_run = jax.random.split(key)
    states, frames = init_envs(k_init, n, level)
    if override is not None:
        p = override(states.params)
        hover = jnp.stack([2.0 / p.twr - 1.0] + [jnp.zeros(n)] * 3, -1)
        states = states._replace(params=p, act_buf=jnp.broadcast_to(hover[:, None], states.act_buf.shape))
    d0, targets = states.prev_dist, states.target

    stateful = hasattr(pilot, "init")

    def one(carry, key):
        states, frames, active, mem = carry
        if stateful:
            action, gaze, mem = pilot.act(states, frames, mem)
        else:
            action, gaze = pilot(states, frames)
        action = jnp.clip(action, -1, 1)
        if gaze is not None:
            states = states._replace(gaze=gaze)
        rec = None
        if log:
            seen = jax.vmap(lambda f, d: f[d])(frames, states.params.cam_delay)
            rec = (states.pos, jnp.linalg.norm(states.vel, axis=-1), seen, action, active)
        states, frames, _, done, info = env_step(key, states, frames, action, level)
        if stateful:
            mem = jnp.where(done[:, None], 0.0, mem)
        out = {k: info[k] & active for k in ("success", "crashed", "timeout")}
        out["time"] = jnp.where(active & done, info["time"], 0.0)
        out["end_dist"] = jnp.where(active & done, info["dist"], 0.0)
        out["speed"] = jnp.where(active, info["speed"], 0.0)
        # distance after the step, before any auto-reset (prev_dist of a reset state is the new episode's)
        out["dist"] = jnp.where(active & ~done, states.prev_dist, jnp.where(info["success"] & active, 0.0, jnp.inf))
        return (states, frames, active & ~done, mem), (rec, out)

    _, (logs, outs) = jax.lax.scan(one, (states, frames, jnp.ones(n, bool), pilot.init(n) if stateful else None),
                                   jax.random.split(k_run, sim.max_steps))
    o = {k: outs[k].any(0) for k in ("success", "crashed", "timeout")}
    o["time"] = outs["time"].sum(0)
    o["peak_speed"] = outs["speed"].max(0)
    o["min_dist"] = outs["dist"].min(0)
    o["end_dist"] = outs["end_dist"].sum(0)
    o["d0"] = d0
    return o, logs, targets


def summarise(o):
    o = jax.device_get(o)
    s = o["success"]
    return {
        "success": float(s.mean()), "crash": float(o["crashed"].mean()), "timeout": float(o["timeout"].mean()),
        "time_to_target": float(o["time"][s].mean()) if s.any() else None,
        "avg_speed": float((o["d0"][s] / o["time"][s]).mean()) if s.any() else None,
        "peak_speed": float(o["peak_speed"][s].mean()) if s.any() else None,
        "n": int(s.size),
        # failed episodes: how close did they get? (near miss = passed within 1.5 m)
        "fail_min_dist_median": float(np.median(o["min_dist"][~s])) if (~s).any() else None,
        "fail_near_miss": float((o["min_dist"][~s] < 1.5).mean()) if (~s).any() else None,
    }


def fmt(name, r):
    t = f"{r['time_to_target']:.2f}s" if r["time_to_target"] else "  -  "
    v = f"{r['avg_speed']:.1f}/{r['peak_speed']:.1f} m/s" if r["avg_speed"] else "-"
    return (f"{name:<26} success {r['success']:6.1%}  crash {r['crash']:6.1%}  timeout {r['timeout']:6.1%}"
            f"  t {t}  avg/peak {v}")


def dump(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {RES}/{name}.json")


# ---------------------------------------------------------------- main
def main_eval(args):
    pilots = {"teacher (true state)": "ckpt/teacher.pkl", "student (camera only)": "ckpt/student.pkl"}
    key = jax.random.PRNGKey(123)   # same key -> same drones, targets and noise for every pilot
    results, logs = {}, {}
    for dr in (1.0, 0.0):
        for name, path in pilots.items():
            out = jax.jit(lambda k: run(make_pilot(path), k, args.episodes, dr, log=dr == 1.0))(key)
            results[f"{name} | DR={dr:g}"] = summarise(out[0])
            print(fmt(f"{name.split()[0]} DR={dr:g}", results[f"{name} | DR={dr:g}"]))
            if dr == 1.0:
                logs[name] = jax.device_get(out)
    dump("main", results)
    plot_flights(logs)
    fpv_gif(logs["student (camera only)"], "student (camera only)")


def plot_flights(results, n_show=6):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for (name, (o, logs, targets)), ls in zip(results.items(), ["-", "--"]):
        pos, speed, _, _, active = logs
        for i in range(n_show):
            m, c = active[:, i], f"C{i}"
            axes[0].plot(*(pos[m, i, :2] - pos[0, i, :2]).T, ls, c=c, lw=1.3)
            axes[0].scatter(*(targets[i, :2] - pos[0, i, :2]), c=c, marker="*", s=120)
            t = np.arange(m.sum()) * EVAL_SIM.ctrl_dt
            axes[1].plot(t, np.linalg.norm(pos[m, i] - targets[i], axis=-1), ls, c=c, lw=1)
            axes[2].plot(t, speed[m, i], ls, c=c, lw=1)
    axes[0].set(title="Top-down paths (solid = teacher, dashed = student)", xlabel="x (m)", ylabel="y (m)",
                aspect="equal")
    axes[1].set(title="Distance to target", xlabel="time (s)", ylabel="m")
    axes[2].set(title="Speed", xlabel="time (s)", ylabel="m/s")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG}/flights.png", dpi=130)
    fig.savefig(f"{FIG}/flights.svg")
    fig.savefig(f"{FIG}/flights.pdf")
    print(f"saved {FIG}/flights.png")


def fpv_gif(result, name):
    o, (pos, speed, frames, action, active), targets = result
    i = int(np.argmax(o["success"]))
    steps = np.nonzero(active[:, i])[0]
    fig, (ax_cam, ax_map) = plt.subplots(1, 2, figsize=(9, 3.6))
    frames = frames[:, :, 0]   # widest view
    im = ax_cam.imshow(frames[0, i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_cam.set_axis_off()
    txt = ax_cam.set_title("", fontsize=8)
    rel = pos[:, i, :2] - pos[0, i, :2]
    ax_map.plot(*rel[steps].T, c="0.7")
    ax_map.scatter(*(targets[i, :2] - pos[0, i, :2]), c="r", marker="*", s=150)
    dot, = ax_map.plot([], [], "o", c="C0")
    ax_map.set(aspect="equal", title="top-down")

    def frame(k):
        im.set_data(frames[k, i])
        txt.set_text(f"{name}: t={k * EVAL_SIM.ctrl_dt:.2f}s  speed={speed[k, i]:.1f} m/s\n"
                     f"thr {action[k, i, 0]:+.2f} roll {action[k, i, 1]:+.2f} "
                     f"pitch {action[k, i, 2]:+.2f} yaw {action[k, i, 3]:+.2f}")
        dot.set_data([rel[k, 0]], [rel[k, 1]])
        return im, txt, dot

    FuncAnimation(fig, frame, frames=steps).save(f"{FIG}/fpv_view.gif", writer=PillowWriter(fps=10))
    print(f"saved {FIG}/fpv_view.gif")


# ---------------------------------------------------------------- robustness sweep
def _wind(p, v):
    n = jnp.linalg.norm(p.wind_mean, axis=-1, keepdims=True)
    direction = jnp.where(n > 1e-6, p.wind_mean / jnp.maximum(n, 1e-6), jnp.array([1.0, 0.0, 0.0]))
    return p._replace(wind_mean=direction * v)


SWEEPS = {
    # name: (values, override(params, v), training range, x label)
    "twr": ([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0],
            lambda p, v: p._replace(twr=jnp.full_like(p.twr, v)), (2.5, 5.0), "thrust-to-weight ratio"),
    "latency": ([0, 1, 2, 3, 4],
                lambda p, v: p._replace(act_delay=jnp.full_like(p.act_delay, v), cam_delay=jnp.full_like(p.cam_delay, v)),
                (0, 2), "control & video latency (steps of 20 ms, each)"),
    "wind": ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], _wind, (0.0, 3.0), "steady wind (m/s)"),
    "pix_noise": ([0.0, 0.025, 0.05, 0.1, 0.15, 0.2],
                  lambda p, v: p._replace(pix_noise=jnp.full_like(p.pix_noise, v)), (0.01, 0.05), "pixel noise σ"),
    "cam_tilt": ([10.0, 20.0, 30.0, 40.0, 50.0],
                 lambda p, v: p._replace(cam_tilt=jnp.full_like(p.cam_tilt, jnp.deg2rad(v))), (20, 40),
                 "camera uptilt (deg)"),
}


def sweep_eval(args):
    pilots = {"teacher (true state)": "ckpt/teacher.pkl", "student (camera only)": "ckpt/student.pkl"}
    key = jax.random.PRNGKey(7)
    results = {}
    for sname, (values, override, _, _) in SWEEPS.items():
        results[sname] = {}
        for pname, path in pilots.items():
            pilot = make_pilot(path)
            f = jax.jit(lambda k, v: run(pilot, k, args.episodes, 1.0, lambda p: override(p, v))[0])
            results[sname][pname] = [summarise(f(key, v)) | {"value": v} for v in values]
            print(sname, pname, [f"{r['success']:.2f}" for r in results[sname][pname]])
    dump("sweep", results)
    plot_sweep(results)


def plot_sweep(results):
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(4 * len(SWEEPS), 3.4), sharey=True)
    for ax, (sname, (values, _, rng, xlabel)) in zip(axes, SWEEPS.items()):
        ax.axvspan(*rng, color="0.9", label="training range")
        for (pname, rows), marker in zip(results[sname].items(), ["o", "s"]):
            ax.plot(values, [r["success"] for r in rows], marker=marker, label=pname)
        ax.set(xlabel=xlabel, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_robustness.png", dpi=150)
    fig.savefig(f"{FIG}/fig6_robustness.svg")
    fig.savefig(f"{FIG}/fig6_robustness.pdf")
    print(f"saved {FIG}/fig6_robustness.png")


# ---------------------------------------------------------------- ablations
def ablation_eval(args):
    key = jax.random.PRNGKey(123)
    results = {}
    for path in sorted(p for p in glob.glob("ckpt/*.pkl")
                       if not os.path.basename(p).startswith("lr_") and "imu" not in os.path.basename(p)):
        name = os.path.basename(path).removesuffix(".pkl")
        results[name] = summarise(jax.jit(lambda k: run(make_pilot(path), k, args.episodes, 1.0)[0])(key))
        print(fmt(name, results[name]))
    dump("ablations", results)


# ---------------------------------------------------------------- long range
def long_eval(args):
    """Long range: the visibility-limited teacher vs the camera-only student (memory + RL),
    on the 15 s benchmark, broken down by target altitude and distance."""
    pilots = {"teacher (visibility-limited)": "ckpt/lr_teacher_vis.pkl",
              "student (camera only, memory + RL)": "ckpt/main/Brl_s0.pkl"}
    sim = load_ckpt(pilots["student (camera only, memory + RL)"])[1]._replace(
        level_mix=False, max_act_delay=2, max_cam_delay=2, max_steps=750, sag_floor=0.0)
    key = jax.random.PRNGKey(321)
    results, logs = {}, {}
    for name, path in pilots.items():
        out = jax.device_get(jax.jit(lambda k: run(make_pilot(path), k, args.episodes, 1.0, sim=sim, log=True))(key))
        o, _, targets = out
        r = summarise(o)
        alt, d0 = targets[:, 2], o["d0"]
        for lo, hi in ((0, 5), (5, 30), (30, 100)):
            m = (alt >= lo) & (alt < hi)
            r[f"success_alt_{lo}_{hi}"] = float(o["success"][m].mean())
        for lo, hi in ((100, 140), (140, 180), (180, 230)):
            m = (d0 >= lo) & (d0 < hi)
            r[f"success_dist_{lo}_{hi}"] = float(o["success"][m].mean())
        results[name] = r
        logs[name] = out
        print(fmt(name.split()[0], r))
    dump("long_flights", results)
    plot_long_flights(logs, sim)


def plot_long_flights(results, sim, n_show=6):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (name, (o, logs, targets)), ls in zip(results.items(), ["-", "--"]):
        pos, speed, _, _, active = logs
        for i in range(n_show):
            m, c = active[:, i], f"C{i}"
            rel = pos[m, i] - pos[0, i]
            horiz_dir = (targets[i, :2] - pos[0, i, :2]) / np.linalg.norm(targets[i, :2] - pos[0, i, :2])
            axes[0].plot(*rel[:, :2].T, ls, c=c, lw=1.2)
            axes[0].scatter(*(targets[i, :2] - pos[0, i, :2]), c=c, marker="*", s=100)
            axes[1].plot(rel[:, :2] @ horiz_dir, pos[m, i, 2], ls, c=c, lw=1.2)
            axes[1].scatter(np.linalg.norm(targets[i, :2] - pos[0, i, :2]), targets[i, 2], c=c, marker="*", s=100)
            axes[2].plot(np.arange(m.sum()) * sim.ctrl_dt, speed[m, i], ls, c=c, lw=1)
    axes[0].set(title="Top-down (solid = teacher, dashed = student)", xlabel="m", ylabel="m", aspect="equal")
    axes[1].set(title="Side view along the start-target line", xlabel="horizontal distance (m)", ylabel="altitude (m)")
    axes[2].set(title="Speed", xlabel="time (s)", ylabel="m/s")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/lr_flights.{ext}", dpi=130)
    print(f"saved {FIG}/lr_flights.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["main", "sweep", "ablations", "long"])
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()
    args.episodes = args.episodes or {"main": 1024, "sweep": 256, "ablations": 1024, "long": 512}[args.mode]
    os.makedirs(FIG, exist_ok=True)
    {"main": main_eval, "sweep": sweep_eval, "ablations": ablation_eval, "long": long_eval}[args.mode](args)
