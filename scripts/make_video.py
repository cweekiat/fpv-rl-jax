"""Short submission video: what the camera-only pilot sees and how it flies, plus results.

    uv run --with "imageio[ffmpeg]" python scripts/make_video.py
writes docs/video/fpv_rl_ral.mp4 (1280x720, 25 fps, ~1 min)
"""
import os
import sys

import imageio.v2 as imageio
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import make_pilot, run  # noqa: E402
from train import load_ckpt  # noqa: E402

FPS, W, H, DPI = 25, 1280, 720, 100
OUT = "docs/video/fpv_rl_ral.mp4"
STUDENT, TEACHER = "ckpt/main/Brl_s0.pkl", "ckpt/lr_teacher_vis.pkl"
DARK, FG, ACCENT = "#101418", "#e8eaed", "#4cc38a"


def canvas():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=DARK)
    return fig


def grab(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def text_card(title, lines=(), seconds=3.0, image=None):
    fig = canvas()
    fig.text(0.5, 0.86 if image else 0.62, title, ha="center", va="center", color=FG, fontsize=30, weight="bold")
    for k, line in enumerate(lines):
        fig.text(0.5, (0.78 if image else 0.48) - 0.065 * k, line, ha="center", va="center", color=FG, fontsize=17)
    if image:
        ax = fig.add_axes([0.06, 0.04, 0.88, 0.62 if lines else 0.72])
        ax.imshow(plt.imread(image))
        ax.axis("off")
    frame = grab(fig)
    return [frame] * int(seconds * FPS)


def rollouts(sim, n=24):
    out = {}
    for name, path in (("student", STUDENT), ("teacher", TEACHER)):
        out[name] = jax.device_get(jax.jit(lambda k: run(make_pilot(path), k, n, 1.0, sim=sim, log=True))(
            jax.random.PRNGKey(2024)))
    return out


def flight_frames(data, i, sim, label, slow=0.6):
    o, (pos, speed, views, action, active), targets = data["student"]
    _, (tpos, _, _, _, tactive), _ = data["teacher"]
    steps = np.nonzero(active[:, i])[0]
    tsteps = np.nonzero(tactive[:, i])[0]
    p0, tgt = pos[0, i], targets[i]
    axis_dir = (tgt[:2] - p0[:2]) / np.linalg.norm(tgt[:2] - p0[:2])
    along = lambda p: (p[..., :2] - p0[:2]) @ axis_dir
    frames = []
    stride = max(1, int(round(FPS / (50 * slow)))) if slow < 1 else 2
    show = steps[:: max(1, int(round(50 / (FPS / slow))))] if slow >= 1 else steps
    every = max(1, int(round(50 * slow / FPS)))      # sim steps per video frame
    for k in steps[::every]:
        fig = canvas()
        fig.text(0.02, 0.955, label, color=FG, fontsize=15, weight="bold")
        fig.text(0.98, 0.955, f"t = {k * sim.ctrl_dt:4.1f} s   speed {speed[k, i]:4.1f} m/s   "
                 f"distance {np.linalg.norm(pos[k, i] - tgt):5.1f} m", color=FG, fontsize=14, ha="right")
        for v, name in enumerate(("120° wide view", "30° gaze view", "8° gaze view")):
            ax = fig.add_axes([0.02 + v * 0.325, 0.47, 0.31, 0.44])
            ax.imshow(views[k, i, v], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(name + " (32×24)", color=FG, fontsize=11)
            ax.axis("off")
        # top-down
        ax = fig.add_axes([0.05, 0.07, 0.27, 0.32], facecolor="#1b2127")
        ax.plot(*(tpos[tsteps, i, :2] - p0[:2]).T, c="0.55", lw=1.2, label="teacher")
        ax.plot(*(pos[steps[0]:k + 1, i, :2] - p0[:2]).T, c=ACCENT, lw=2, label="camera-only student")
        ax.scatter(*(tgt[:2] - p0[:2]), marker="*", s=180, c="#ff6b6b", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("top-down (m)", color=FG, fontsize=10)
        ax.tick_params(colors=FG, labelsize=7)
        ax.legend(fontsize=7, loc="best", facecolor="#1b2127", labelcolor=FG)
        # side view
        ax = fig.add_axes([0.38, 0.07, 0.27, 0.32], facecolor="#1b2127")
        ax.plot(along(tpos[tsteps, i]), tpos[tsteps, i, 2], c="0.55", lw=1.2)
        ax.plot(along(pos[steps[0]:k + 1, i]), pos[steps[0]:k + 1, i, 2], c=ACCENT, lw=2)
        ax.scatter(np.linalg.norm(tgt[:2] - p0[:2]), tgt[2], marker="*", s=180, c="#ff6b6b", zorder=3)
        ax.set_title("side view: altitude vs distance (m)", color=FG, fontsize=10)
        ax.tick_params(colors=FG, labelsize=7)
        # sticks
        ax = fig.add_axes([0.72, 0.07, 0.25, 0.32], facecolor="#1b2127")
        ax.barh(["yaw", "pitch", "roll", "throttle"], action[k, i, ::-1], color=ACCENT)
        ax.set_xlim(-1, 1)
        ax.axvline(0, c="0.5", lw=0.8)
        ax.set_title("stick commands", color=FG, fontsize=10)
        ax.tick_params(colors=FG, labelsize=8)
        frames.append(grab(fig))
    return frames + [frames[-1]] * int(1.0 * FPS)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sim = load_ckpt(STUDENT)[1]._replace(level_mix=False, max_act_delay=2, max_cam_delay=2, max_steps=750, sag_floor=0.0)
    data = rollouts(sim)
    o = data["student"][0]
    t = o["time"]
    good = [i for i in np.argsort(-o["d0"]) if o["success"][i] and 4.0 < t[i] < 10.0][:2]
    frames = []
    frames += text_card("From Pixels to Sticks", ["What a camera-only FPV acro pilot needs to learn",
                                                  "Wee Kiat Chan"], 3.5)
    frames += text_card("The pilot sees only video", [
        "Inputs: three delayed, noisy 32×24 views and its own past stick commands",
        "Outputs: throttle and roll / pitch / yaw rates (acro mode), plus where to look",
        "No position, velocity or attitude estimate"], 5.0)
    frames += text_card("Teacher → student", [], 5.0, image="docs/figures/fig1_overview.png")
    for n, i in enumerate(good):
        frames += text_card(f"Flight {n + 1}: target {o['d0'][i]:.0f} m away",
                            ["camera-only student (memory + RL), shown at 0.6× speed"], 2.0)
        frames += flight_frames(data, i, sim, f"Camera-only student, flight {n + 1}")
    frames += text_card("What the pilot needs (5 seeds)", [], 6.0, image="docs/figures/fig12_main.png")
    frames += text_card("Why memory helps", [
        "Linear probes: attitude is already in the current frames,",
        "but only the memory recovers velocity (R² 0.57 → 0.79).",
        "Best student: 77% of 100–200 m targets at up to 40 m/s;",
        "a calibrated hand-designed visual-servo pilot: 3%."], 6.0)
    frames += text_card("Simulation only · single laptop CPU · pure JAX",
                        ["github.com/cweekiat/fpv-rl-jax"], 3.5)
    imageio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                     ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved {OUT}: {len(frames) / FPS:.0f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
