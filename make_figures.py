"""Generate the explanatory figures in docs/figures/.

    uv run python make_figures.py
"""
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

from fpv import camera, env
from fpv.nets import init_cnn

OUT = "docs/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path"})

C = {  # one colour per role, used consistently across figures
    "sim": "#e8ecef", "teacher": "#cfe2f3", "student": "#fce5cd", "critic": "#e4d7f5",
    "update": "#fff2cc", "camera": "#d9ead3", "dr": "#f4e1ea", "priv": "#c0392b", "edge": "#333333",
}


# ---------------------------------------------------------------- drawing helpers
def box(ax, x, y, w, h, text, fc, fs=9, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=C["edge"], lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
            linespacing=1.35)


def flow(ax, pts, label=None, at=None, color=C["edge"], ls="-", fs=8, ha="center", va="bottom",
         offset=(0, 0.08)):
    """Polyline with an arrowhead on the last segment. Label goes at segment `at`'s midpoint."""
    pts = np.asarray(pts, float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=1.3, ls=ls, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=12))
    if label:
        i = len(pts) - 2 if at is None else at
        mid = (pts[i] + pts[i + 1]) / 2 + np.array(offset)
        ax.text(*mid, label, fontsize=fs, ha=ha, va=va, color=color if color != C["edge"] else "black")


def canvas(ax, xlim, ylim, title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, weight="bold", loc="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print(f"saved {OUT}/{name}.svg/.png")


# ---------------------------------------------------------------- figure 1
def fig_overview():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [10.4, 11.2, 6.2]})

    ax = axes[0]
    canvas(ax, (0, 10.4), (0, 7), "(a) Stage 1: privileged teacher (PPO)")
    box(ax, 0.3, 0.5, 3.6, 2.2, "Simulator\n1024 parallel drones\n+ domain rand. $\\xi\\sim p(\\xi)$", C["sim"])
    box(ax, 5.6, 4.4, 3.8, 1.5, "Teacher  $\\pi_T(a\\,|\\,s)$\nMLP 256-256", C["teacher"], weight="bold")
    box(ax, 5.6, 1.9, 3.8, 1.3, "Critic  $V(s,\\xi,t_{left})$\nMLP 256-256", C["critic"])
    box(ax, 5.6, 0.2, 3.8, 1.1, "PPO update\nGAE $\\lambda$=0.95, clip 0.2", C["update"])
    flow(ax, [(1.3, 2.7), (1.3, 5.15), (5.6, 5.15)], "privileged state $s$ (25-d)", at=1, color=C["priv"])
    flow(ax, [(5.6, 4.7), (3.6, 4.7), (3.6, 2.7)], "sticks $a$", at=0, offset=(0, -0.38))
    flow(ax, [(3.9, 2.55), (5.6, 2.55)], "$s,\\xi,t$", color=C["priv"])
    flow(ax, [(3.9, 0.75), (5.6, 0.75)], "$r$, done")
    flow(ax, [(9.4, 0.75), (9.95, 0.75), (9.95, 5.15), (9.4, 5.15)], ls="--")
    flow(ax, [(9.95, 2.55), (9.4, 2.55)], ls="--")
    ax.text(10.05, 3.9, "$\\nabla\\theta$", fontsize=9)
    ax.text(7.5, 1.55, "discarded after training", fontsize=7.5, style="italic", ha="center", color="0.35")

    ax = axes[1]
    canvas(ax, (0, 11.2), (-0.2, 7), "(b) Stage 2: camera-only student (DAgger)")
    box(ax, 0.3, 0.6, 3.1, 2.1, "Simulator\n512 parallel drones\n+ domain rand.", C["sim"])
    box(ax, 0.3, 4.6, 3.1, 1.5, "FPV camera\nray-cast 32×24\nnoise, delay, freezes", C["camera"], fs=8.5)
    box(ax, 5.0, 4.6, 3.0, 1.5, "Student  $\\pi_S$\nCNN + MLP", C["student"], weight="bold")
    box(ax, 5.0, 1.6, 3.0, 1.2, "Teacher  $\\pi_T$\n(frozen)", C["teacher"], ls="--")
    box(ax, 5.3, 3.2, 2.4, 0.95, "$\\mathcal{L}=\\|a_S-a_T\\|^2$", C["update"])
    ax.add_patch(Circle((9.3, 3.7), 0.42, fc="white", ec=C["edge"], lw=1.1))
    ax.text(9.3, 3.7, "$\\beta$", ha="center", va="center", fontsize=12)
    flow(ax, [(1.85, 2.7), (1.85, 4.6)], "pose", offset=(0.12, 0), ha="left", va="center")
    flow(ax, [(3.4, 5.35), (5.0, 5.35)], "$o$: 3 frames")
    flow(ax, [(3.4, 2.2), (5.0, 2.2)], "$s$", color=C["priv"])
    flow(ax, [(6.5, 4.6), (6.5, 4.15)])
    flow(ax, [(6.5, 2.8), (6.5, 3.2)])
    ax.text(6.6, 2.92, "label", fontsize=7.5, color="0.35")
    flow(ax, [(8.0, 5.35), (9.3, 5.35), (9.3, 4.12)], "$a_S$", at=0)
    flow(ax, [(8.0, 2.2), (9.3, 2.2), (9.3, 3.28)], "$a_T$", at=0, offset=(0, -0.35))
    flow(ax, [(9.72, 3.7), (10.6, 3.7), (10.6, 0.15), (1.85, 0.15), (1.85, 0.6)],
         "executed action", at=2, offset=(0, 0.06))
    ax.text(9.9, 4.35, "$a_T$ w.p. $\\beta$\n$\\beta$: 1→0", fontsize=7.5, color="0.3")

    ax = axes[2]
    canvas(ax, (0, 6.2), (0, 7), "(c) Deployment")
    box(ax, 0.4, 0.6, 5.4, 1.6, "Drone\n(simulated, or real\nwith an FC in acro mode)", C["sim"])
    box(ax, 0.4, 3.4, 2.0, 1.5, "FPV\ncamera", C["camera"])
    box(ax, 3.8, 3.4, 2.0, 1.5, "Student\n$\\pi_S$", C["student"], weight="bold")
    flow(ax, [(1.4, 2.2), (1.4, 3.4)])
    flow(ax, [(2.4, 4.15), (3.8, 4.15)], "video")
    flow(ax, [(4.8, 3.4), (4.8, 2.2)], "sticks", offset=(0.1, 0), ha="left", va="center")
    ax.text(3.1, 5.9, "no state estimator\nno privileged information", ha="center", fontsize=8.5,
            style="italic", color="0.3")
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- figure 2
def fig_simulator():
    fig, ax = plt.subplots(figsize=(15, 6.6))
    canvas(ax, (0, 14.2), (0, 6.6))
    box(ax, 0.1, 1.95, 1.9, 1.7, "Policy\n$\\pi$\n50 Hz", C["student"], weight="bold")
    # actuation path (top row)
    box(ax, 2.6, 3.6, 2.1, 1.4, "Latency\nbuffer*\n0–40 ms", C["sim"])
    box(ax, 5.2, 3.6, 2.5, 1.4, "Flight controller*\nrate tracking $\\tau_{rate}$\ntorque limit · headroom", C["sim"])
    box(ax, 8.2, 3.6, 2.1, 1.4, "Motors*\nspin-up $\\tau_{mot}$\nbattery sag", C["sim"])
    box(ax, 10.8, 3.6, 3.3, 1.4, "Rigid body, 100 Hz*\ngravity · drag on $v-w$\nwind + gusts · ground effect\nthrust & vibration noise",
        C["sim"], fs=8.5)
    # perception path (bottom row)
    box(ax, 10.8, 0.5, 3.3, 1.5, "Ray-cast camera*\n120° FOV, uptilt $\\theta$\nsky · checker ground · target", C["camera"], fs=8.5)
    box(ax, 7.6, 0.5, 2.6, 1.5, "Video link*\nexposure · pixel noise\nlatency · frame freezes", C["camera"], fs=8.5)
    box(ax, 4.8, 0.5, 2.2, 1.5, "Frame stack\n3 × 24×32\n+ stick history", C["camera"], fs=8.5)

    flow(ax, [(1.05, 3.65), (1.05, 4.3), (2.6, 4.3)], "sticks $a_t$", at=1)
    flow(ax, [(4.7, 4.3), (5.2, 4.3)])
    flow(ax, [(7.7, 4.3), (8.2, 4.3)])
    flow(ax, [(10.3, 4.3), (10.8, 4.3)])
    flow(ax, [(12.45, 3.6), (12.45, 2.0)], "pose $(p, q)$", offset=(0.12, 0), ha="left", va="center")
    flow(ax, [(10.8, 1.25), (10.2, 1.25)])
    flow(ax, [(7.6, 1.25), (7.0, 1.25)])
    flow(ax, [(4.8, 1.25), (1.05, 1.25), (1.05, 1.95)], "observation $o_t$", at=0)
    flow(ax, [(11.3, 3.6), (11.3, 2.75), (2.0, 2.75)], "privileged state $s_t$ (teacher & critic only, never the student)",
         at=1, color=C["priv"], ls="--")
    # domain randomisation
    box(ax, 3.2, 5.55, 10.0, 0.8, "Domain randomisation   $\\xi \\sim p(\\xi)$, resampled every episode "
        "(drone, world, camera, link: 18 parameters)", C["dr"])
    for x in (3.65, 6.45, 9.25, 12.45):
        flow(ax, [(x, 5.55), (x, 5.0)], color="#8e44ad", ls=":")
    ax.text(14.1, 0.1, "* depends on randomised parameters", fontsize=7.5, ha="right", style="italic", color="0.35")
    save(fig, "fig2_simulator")


# ---------------------------------------------------------------- figure 3
def axis_angle(axis, deg):
    axis = jnp.asarray(axis, float) / jnp.linalg.norm(jnp.asarray(axis, float))
    a = jnp.deg2rad(deg)
    return jnp.concatenate([jnp.cos(a / 2)[None], jnp.sin(a / 2) * axis])


def fig_camera():
    base = env.reset(jax.random.PRNGKey(0), dr_scale=0.0)
    clean = base.params._replace(cam_tilt=jnp.deg2rad(30.0), tile=jnp.array(1.0), gain=jnp.array(1.0),
                                 offset=jnp.array(0.0), pix_noise=jnp.array(0.02))
    level = jnp.array([1.0, 0, 0, 0])
    tgt = jnp.array([8.0, 1.0, 2.5])

    def state(pos=(0, 0, 2.0), quat=level, **p):
        return base._replace(pos=jnp.array(pos, float), quat=quat, target=tgt, params=clean._replace(**p))

    hi = env.Sim(img_h=192, img_w=256)
    lo = env.Sim()
    key = jax.random.PRNGKey(3)
    scenes = [
        ("hover, target 8 m ahead", state()),
        ("rolled 30°", state(quat=axis_angle([1, 0, 0], 30))),
        ("pitched 35° nose-down\n(fast forward flight)", state(quat=axis_angle([0, 1, 0], 35))),
        ("1.5 m from target", state(pos=(6.5, 0.8, 2.3))),
    ]
    sensors = [
        ("clean sensor (no noise)", state(pix_noise=jnp.array(0.0))),
        ("dark exposure + heavy noise", state(gain=jnp.array(0.8), offset=jnp.array(-0.05), pix_noise=jnp.array(0.05))),
        ("bright exposure", state(gain=jnp.array(1.2), offset=jnp.array(0.05))),
        ("uptilt 40° instead of 30°", state(cam_tilt=jnp.deg2rad(40.0))),
    ]
    # Frame stack while flying forward at 10 m/s (frames 20 ms apart).
    fwd = axis_angle([0, 1, 0], 30)
    stack = [state(pos=(0.2 * k, 0, 1.2), quat=fwd) for k in (2, 1, 0)]

    fig, axes = plt.subplots(4, 4, figsize=(13, 10.2))
    for j, (name, s) in enumerate(scenes):
        axes[0, j].imshow(camera.render(key, s._replace(params=s.params._replace(pix_noise=jnp.array(0.0))), hi)[0],
                          cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name, fontsize=9)
        axes[1, j].imshow(camera.render(key, s, lo)[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for j, (name, s) in enumerate(sensors):
        axes[2, j].imshow(camera.render(key, s, lo)[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[2, j].set_title(name, fontsize=9)
    frames = [camera.render(key, s, lo)[0] for s in stack]
    for j, (f, name) in enumerate(zip(frames, ["$t$ (newest)", "$t-1$", "$t-2$"])):
        axes[3, j].imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[3, j].set_title(f"frame {name}", fontsize=9)
    axes[3, 3].imshow(np.abs(np.asarray(frames[0]) - np.asarray(frames[2])), cmap="magma", interpolation="nearest")
    axes[3, 3].set_title("|frame t − frame t−2|\nmotion cue = speed", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    rows = ["reference\n(256×192, not used)", "what the pilot gets\n(32×24)", "sensor & camera\nrandomisation",
            "frame stack at 10 m/s"]
    for ax, r in zip(axes[:, 0], rows):
        ax.set_ylabel(r, fontsize=9.5, weight="bold")
    fig.tight_layout()
    save(fig, "fig3_camera")


# ---------------------------------------------------------------- figure 4
def fig_networks():
    fig, ax = plt.subplots(figsize=(15, 6.4))
    canvas(ax, (0, 15.5), (0, 6.6))
    n_params = sum(x.size for x in jax.tree.leaves(
        init_cnn(jax.random.PRNGKey(0), (24, 32, 3), 12, env.ACT_DIM)))
    ax.text(0.1, 6.3, f"Student $\\pi_S$ (camera only), {n_params / 1e3:.0f}k params", fontsize=10.5, weight="bold")
    # input frames as stacked thumbnails
    s = env.reset(jax.random.PRNGKey(5), dr_scale=0.0)
    img = np.asarray(camera.render(jax.random.PRNGKey(1), s)[0])
    for k in range(3):
        x0, y0 = 0.2 + 0.18 * k, 3.6 + 0.18 * (2 - k)
        ax.imshow(img, cmap="gray", extent=(x0, x0 + 1.6, y0, y0 + 1.2), vmin=0, vmax=1, zorder=3 + k,
                  interpolation="nearest")
        ax.add_patch(FancyBboxPatch((x0, y0), 1.6, 1.2, boxstyle="square,pad=0", fc="none", ec="k", lw=0.6,
                                    zorder=3 + k))
    ax.text(1.15, 3.3, "frames\n3×24×32", ha="center", va="top", fontsize=8)
    ax.set_aspect("auto")
    layers = [("Conv 5×5 /2\n16 ch\n→ 12×16×16", 2.3, 1.9), ("Conv 3×3 /2\n32 ch\n→ 6×8×32", 4.2, 1.5),
              ("Conv 3×3 /2\n32 ch\n→ 3×4×32", 6.1, 1.15), ("flatten\n384", 7.9, 0.8)]
    cy = 4.45
    prev = 2.1
    for text, x, h in layers:
        flow(ax, [(prev, cy), (x, cy)])
        box(ax, x, cy - h / 2, 1.5, h, text, C["student"], fs=8)
        prev = x + 1.5
    ax.add_patch(Circle((9.95, cy), 0.28, fc="white", ec=C["edge"]))
    ax.text(9.95, cy, "⊕", ha="center", va="center", fontsize=13)
    flow(ax, [(prev, cy), (9.67, cy)])
    box(ax, 9.25, 2.55, 1.4, 0.8, "stick history\n3×4 = 12", C["camera"], fs=8)
    flow(ax, [(9.95, 3.35), (9.95, cy - 0.28)])
    prev = 10.23
    for text, x in [("FC 256\nReLU", 10.7), ("FC 256\nReLU", 12.2)]:
        flow(ax, [(prev, cy), (x, cy)])
        box(ax, x, cy - 0.55, 1.1, 1.1, text, C["student"], fs=8)
        prev = x + 1.1
    flow(ax, [(prev, cy), (13.75, cy)])
    box(ax, 13.75, cy - 0.65, 1.6, 1.3, "4 sticks\nthr, $\\omega_x$,\n$\\omega_y$, $\\omega_z$", C["update"], fs=8)

    ax.plot([0.1, 15.4], [2.25, 2.25], color="0.8", lw=0.8)
    for y0, name, inp, act, out, fc in [
        (1.35, "Teacher $\\pi_T$ (actor)", "$s$: 25\nprivileged", "tanh", "$\\mu$: 4 sticks\n+ learned $\\log\\sigma$", C["teacher"]),
        (0.2, "Critic $V$", "$s,\\xi,t$: 37\n+ hidden params", "tanh", "value: 1", C["critic"])]:
        ax.text(0.1, y0 + 0.4, name, fontsize=10, weight="bold", va="center")
        box(ax, 3.2, y0, 1.9, 0.8, inp, C["sim"], fs=8)
        prev = 5.1
        for x in (6.0, 7.9):
            flow(ax, [(prev, y0 + 0.4), (x, y0 + 0.4)])
            box(ax, x, y0, 1.3, 0.8, f"FC 256\n{act}", fc, fs=8)
            prev = x + 1.3
        flow(ax, [(prev, y0 + 0.4), (10.2, y0 + 0.4)])
        box(ax, 10.2, y0, 2.0, 0.8, out, C["update"], fs=8)
    ax.text(15.4, 0.6, "Convs are im2col + matmul\n(fast inside lax.scan on CPU)", ha="right", fontsize=7.5,
            style="italic", color="0.35")
    save(fig, "fig4_networks")


# ---------------------------------------------------------------- figure 5
def drone_icon(ax, x, z, pitch_deg, size=0.9):
    """Side view; positive pitch = nose down."""
    a = np.deg2rad(-pitch_deg)
    d = np.array([np.cos(a), np.sin(a)]) * size / 2
    ax.plot([x - d[0], x + d[0]], [z - d[1], z + d[1]], color="k", lw=3, solid_capstyle="round", zorder=5)
    n = np.array([-np.sin(a), np.cos(a)]) * 0.12
    for sgn in (-1, 1):
        cx, cz = np.array([x, z]) + sgn * d + n
        ax.plot([cx - d[0] * 0.5, cx + d[0] * 0.5], [cz - d[1] * 0.5, cz + d[1] * 0.5], color="0.4", lw=1.5, zorder=5)
    return a


def fig_task():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.05, 1, 0.42]})

    # (a) top-down
    ax1.set_title("(a) Top-down: start and target sampling", fontsize=11, weight="bold", loc="left")
    ax1.add_patch(Wedge((0, 0), 12, -50, 50, width=8, fc=C["dr"], ec="#8e44ad", lw=1, alpha=0.7))
    ax1.add_patch(Wedge((0, 0), 13.5, -60, 60, fc="none", ec="0.6", lw=0.8, ls="--"))
    ax1.text(0.3, 7.2, "camera FOV ±60°", fontsize=8, color="0.4")
    ax1.text(6.6, -1.35, "target spawn region\n4–12 m, ±50° of heading", fontsize=8.5, ha="center", color="#6c3483")
    ax1.add_patch(Polygon([[0.0, 0.5], [0.0, -0.5], [1.2, 0.0]], fc="k"))
    ax1.text(-0.3, -0.9, "drone start\nrandom heading", fontsize=8, ha="center", va="top")
    tx, ty = 8.5, 3.6
    t = np.linspace(0, 1, 50)
    ax1.plot(1.2 + (tx - 1.2) * t, 1.8 * np.sin(np.pi * t) * 0.3 + ty * t ** 1.3, color="C0", lw=1.6)
    ax1.add_patch(Circle((tx, ty), 0.5, fc="none", ec="C2", lw=1.5))
    ax1.plot(tx, ty, marker="*", ms=14, color="C3")
    ax1.annotate("success radius 0.5 m", (tx + 0.4, ty + 0.35), (tx + 1.0, ty + 2.3), fontsize=8,
                 arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax3.axis("off")
    ax3.set_title("(c) Reward", fontsize=11, weight="bold", loc="left")
    ax3.text(0.0, 0.95, "per 50 Hz step\n"
             "  + 2.0 · progress toward target\n"
             "  − 0.01 time penalty (be fast)\n"
             "  + 0.05 · cos∠(camera axis, target)\n"
             "  − small spin / stick-jerk penalties\n"
             "  + 10 reached,  − 10 crash\n\n"
             "episode ends on reach, crash\n"
             "or 5 s timeout, so the\n"
             "highest return = fastest\n"
             "safe approach", fontsize=8.5, va="top", family="DejaVu Sans Mono", transform=ax3.transAxes,
             bbox=dict(fc="white", ec="0.7", boxstyle="round,pad=0.4"))
    ax1.set(xlim=(-2, 15), ylim=(-10.5, 10.5), aspect="equal", xlabel="m", ylabel="m")
    ax1.grid(alpha=0.2)

    # (b) side view: why FPV cameras are tilted up
    ax2.set_title("(b) Side view: altitudes, and why the camera is tilted up", fontsize=11, weight="bold", loc="left")
    ax2.axhspan(-0.6, 0, color="#a0522d", alpha=0.25)
    ax2.text(6.0, -0.35, "ground: crash (also z > 15 m or > 25 m away)", fontsize=8, ha="center", va="center")
    ax2.add_patch(plt.Rectangle((-0.7, 1), 1.4, 2, fc="C0", alpha=0.12))
    ax2.text(0, 3.15, "start\n1–3 m", fontsize=8, ha="center", va="bottom")
    ax2.add_patch(plt.Rectangle((4, 1), 8, 3, fc=C["dr"], alpha=0.7))
    ax2.text(8, 4.15, "target altitude 1–4 m", fontsize=8, ha="center", va="bottom", color="#6c3483")

    for x, z, pitch, label in [(0.0, 2.0, 0, "hovering:\ncamera looks up 30°"),
                               (6.5, 2.4, 30, "flying fast, pitched 30° nose-down:\ncamera looks level at the target")]:
        a = drone_icon(ax2, x, z, pitch)
        cam = a + np.deg2rad(30)
        half = np.deg2rad(52)
        r = 2.6
        ax2.add_patch(Wedge((x, z), r, np.rad2deg(cam - half), np.rad2deg(cam + half), fc="C1", alpha=0.15, ec="C1"))
        ax2.annotate("", (x + r * np.cos(cam), z + r * np.sin(cam)), (x, z),
                     arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.4))
        ax2.text(x, z - 0.55, label, fontsize=8, ha="center", va="top")
    ax2.plot(11.0, 2.6, marker="*", ms=14, color="C3")
    ax2.text(9.0, 6.4, "uptilt $\\theta$ ∈ [20°, 40°] randomised;\nvertical FOV ≈ 105°", fontsize=8, ha="center",
             va="top", color="C1")
    ax2.set(xlim=(-1.5, 13), ylim=(-0.6, 6.5), aspect="equal", xlabel="horizontal distance (m)", ylabel="altitude (m)")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "fig5_task")


# ---------------------------------------------------------------- figure 8
def fig_foveation():
    """Long-range camera: wide view plus two gaze-steered narrow views of one camera."""
    sim = env.long_range_sim()
    base = env.reset(jax.random.PRNGKey(2), sim, 0.0, 1.0)
    cases = [("hover at 2 m, target 30 m", (0, 0, 2), 0.0, (30, 2, 5)),
             ("hover at 2 m, target 120 m", (0, 0, 2), 0.0, (120, 4, 20)),
             ("pitched 70° (≈40 m/s), target 150 m", (0, 0, 20), 70.0, (150, 3, 20)),
             ("altitude 80 m, target 200 m", (0, 0, 80), 30.0, (200, 3, 60))]
    fig, axes = plt.subplots(len(cases), 3, figsize=(8.4, 7.6))
    for r, (name, pos, pitch, tgt) in enumerate(cases):
        a = np.deg2rad(pitch)
        s = base._replace(pos=jnp.array(pos, float), quat=jnp.array([np.cos(a / 2), 0, np.sin(a / 2), 0]),
                          target=jnp.array(tgt, float), params=base.params._replace(cam_tilt=jnp.deg2rad(45.0)))
        s = s._replace(gaze=camera.target_gaze(s, sim))
        imgs = camera.render(jax.random.PRNGKey(0), s, sim)
        for c, fov in enumerate(sim.views_hfov):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{fov:g}° view" + ("" if c == 0 else " (gaze-steered)"), fontsize=9)
        g = np.asarray(s.gaze)
        axes[r, 0].plot((g[0] + 1) / 2 * sim.img_w - 0.5, (g[1] + 1) / 2 * sim.img_h - 0.5, "+", c="C3", ms=11, mew=1.5)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    save(fig, "fig8_foveation")




# ---------------------------------------------------------------- figure 12
def fig_main():
    """Main experiment: memory x training stage (imitation, + compute-matched DAgger, + RL), 5 seeds."""
    import json
    mj = json.load(open("results/camera_main.json"))
    for extra in ("results/seeds34.json", "results/controls.json"):
        if os.path.exists(extra):
            mj.update(json.load(open(extra)))
    cells = (("A", "imitation"), ("Adg", "+ more\nDAgger"), ("Arl", "+ RL\nfine-tune"),
             ("B", "imitation"), ("Bdg", "+ more\nDAgger"), ("Brl", "+ RL\nfine-tune"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (bench, title) in zip(axes, (("standard_15s", "15 s (standard)"), ("long_30s", "30 s"))):
        for i, (c, label) in enumerate(cells):
            v = np.array([mj[f"ckpt/main/{c}_s{s}.pkl"][bench]["success"] for s in range(5)
                          if f"ckpt/main/{c}_s{s}.pkl" in mj])
            x = i + (0.6 if i >= 3 else 0)
            ax.bar(x, v.mean(), yerr=v.std(ddof=1), capsize=3, color="C7" if c[0] == "A" else "C2",
                   alpha={1: 0.45, 2: 0.7, 3: 1.0}[len(c)] if len(c) < 3 else 1.0)
            ax.scatter([x] * len(v), v, c="k", s=8, zorder=3)
            ax.text(x, min(v.mean() + v.std(ddof=1) + 0.03, 1.02), f"{100 * v.mean():.0f}", ha="center", fontsize=7)
        ax.axhline(0.995, ls="--", c="0.4", lw=1)
        ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], [c[1] for c in cells], fontsize=7)
        ax.text(1, -0.24, "no memory", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(4.6, -0.24, "memory (GRU)", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
        ax.set(title=title, ylim=(0, 1.1))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("long-range success")
    fig.tight_layout()
    save(fig, "fig12_main")


# ---------------------------------------------------------------- figure 13
def fig_training_lr():
    """Long-range DAgger training (5 seeds): success through the teacher hand-over."""
    import re
    pat = re.compile(r"iter\s+(\d+) \|.*?success ([\d.]+) crash ([\d.]+) timeout ([\d.]+).*?mse ([\d.]+)")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for arch, c, label in (("A", "C7", "no memory"), ("B", "C2", "memory (GRU)")):
        runs = []
        for s in range(5):
            rows = [tuple(map(float, m.groups())) for m in map(pat.search, open(f"logs/main/{arch}_s{s}.txt")) if m]
            runs.append(np.array(rows))
        n = min(len(r) for r in runs)
        it = runs[0][:n, 0]
        for ax, col in zip(axes, (1, 4)):
            v = np.stack([r[:n, col] for r in runs])
            ax.plot(it, v.mean(0), c=c, label=label)
            ax.fill_between(it, v.min(0), v.max(0), color=c, alpha=0.2)
    for ax in axes:
        ax.axvline(250, ls="--", c="0.4", lw=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlabel("DAgger iteration")
    axes[0].text(255, 0.05, r"$\beta=0$", fontsize=8, color="0.3")
    axes[0].set(ylabel="training success", ylim=(0, 1.02), title="Success (mean, seed range)")
    axes[1].set(ylabel="imitation MSE", yscale="log", title="Imitation error")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "fig13_training_lr")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_overview()
    fig_simulator()
    fig_camera()
    fig_networks()
    fig_task()
    fig_foveation()
    # result figures need their evaluation JSONs or training logs (see reproduce.sh)
    for f, needs in ((fig_main, "results/seeds34.json"), (fig_training_lr, "logs/main/B_s4.txt")):
        if os.path.exists(needs):
            f()
