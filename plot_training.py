"""Plot training curves from the logs written by train.py, one row per stage.

    uv run python plot_training.py logs/teacher.txt logs/student.txt
"""
import re
import sys

import matplotlib.pyplot as plt

PATTERN = re.compile(r"iter\s+(\d+) \|\s+([\d.]+)M steps \| success ([\d.]+) crash ([\d.]+) "
                     r"timeout ([\d.]+) \| t_target ([\d.]+)s")
BETA_ITERS = 200  # dagger.Config.beta_iters: the teacher stops flying after this


def parse(path):
    rows = [m.groups() for m in map(PATTERN.search, open(path)) if m]
    return [list(map(float, col)) for col in zip(*rows)]


paths = sys.argv[1:]
fig, axes = plt.subplots(len(paths), 2, figsize=(7.2, 2.6 * len(paths)), squeeze=False)
for (ax_o, ax_t), path in zip(axes, paths):
    it, steps, succ, crash, timeout, t_target = parse(path)
    name = path.split("/")[-1].removesuffix(".txt")
    ax_o.plot(steps, succ, label="success", c="C2")
    ax_o.plot(steps, crash, label="crash", c="C3")
    ax_o.plot(steps, timeout, label="timeout", c="C7", lw=0.8)
    ax_t.plot(steps, t_target, c="C0")
    if name.startswith("student"):
        hand_over = steps[min(range(len(it)), key=lambda i: abs(it[i] - BETA_ITERS))]
        for ax in (ax_o, ax_t):
            ax.axvline(hand_over, c="0.5", ls="--", lw=0.8)
        ax_o.text(hand_over, 0.5, r" $\beta=0$", fontsize=8, color="0.4")
    ax_o.set(title=f"{name}: episode outcomes", xlabel="env steps (M)", ylim=(-0.02, 1.02))
    ax_t.set(title=f"{name}: time to target", xlabel="env steps (M)", ylabel="s")
    ax_o.legend(fontsize=7, loc="center right")
    for ax in (ax_o, ax_t):
        ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/figures/training_curves.png", dpi=130)
fig.savefig("docs/figures/training_curves.pdf")
print("saved docs/figures/training_curves.png")
