"""Two-stage training.

    uv run python train.py teacher            # stage 1: PPO on privileged state
    uv run python train.py student            # stage 2: DAgger, camera-only pilot
    uv run python train.py teacher --iters 20 # quick smoke test

Long-range task (targets 100-200 m away, ~40 m/s drone, foveated camera):
    uv run python train.py teacher --preset long --out ckpt/lr_teacher.pkl
    uv run python train.py student --preset long --teacher ckpt/lr_teacher.pkl --out ckpt/lr_student.pkl

Ablations used in the paper:
    uv run python train.py teacher --dr 0 --iters 300 --out ckpt/teacher_nodr.pkl
    uv run python train.py teacher --w-perception 0 --iters 300 --out ckpt/teacher_noperc.pkl
    uv run python train.py student --teacher ckpt/teacher_noperc.pkl --out ckpt/student_noperc.pkl
    uv run python train.py student --frame-stack 1 --out ckpt/student_1frame.pkl
    uv run python train.py student --bc --out ckpt/student_bc.pkl
"""
import argparse
import os
import pickle
import time

import jax

from fpv import dagger, env, finetune, ppo

# Curriculum (only for sims with curriculum=True): raise the difficulty level whenever
# the policy succeeds often enough at the current one.
CURRICULUM_SUCCESS = 0.8
CURRICULUM_STEP = 0.025


class Logger:
    """Prints progress with the throughput of the last interval, which shows thermal
    throttling on a fanless laptop more clearly than a running average."""

    def __init__(self, n, steps_per_iter, every=10):
        self.n, self.spi, self.every = n, steps_per_iter, every
        self.t_last, self.i_last = time.time(), -1

    def __call__(self, i, m, level=None):
        if i % self.every and i != self.n - 1:
            return
        now = time.time()
        rate = (i - self.i_last) * self.spi / (now - self.t_last)
        self.t_last, self.i_last = now, i
        extra = f"mse {float(m['imitation_mse']):.4f}" if "imitation_mse" in m else \
            f"reward {float(m['reward']):6.3f}"
        lv = f" | level {level:.3f}" if level is not None else ""
        if "gaze_err" in m and float(m["gaze_err"]) > 0:
            lv += f" | gaze err {float(m['gaze_err']):.3f}"
        print(f"iter {i:4d} | {(i + 1) * self.spi / 1e6:6.1f}M steps | success {float(m['success']):.2f} "
              f"crash {float(m['crash']):.2f} timeout {float(m['timeout']):.2f} | "
              f"t_target {float(m['time_to_target']):.2f}s | {extra}{lv} | {rate / 1e3:.0f}k steps/s", flush=True)


class Curriculum:
    def __init__(self, sim, start=0.0):
        self.on = sim.curriculum
        self.level = start if self.on else 1.0

    def update(self, m):
        if self.on and float(m["success"]) >= CURRICULUM_SUCCESS:
            self.level = min(1.0, self.level + CURRICULUM_STEP)
        return self.level


def load_ckpt(path):
    """Returns (params, sim). Also reads bare-params checkpoints (default Sim)."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return ckpt["params"], env.Sim(**ckpt["sim"])
    return ckpt, env.Sim()


def save_ckpt(path, params, sim, args, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"params": jax.device_get(params), "sim": sim._asdict(), "args": vars(args), **extra}, f)
    os.replace(path + ".tmp", path)   # atomic, so a crash mid-save never leaves a broken file


def widen_inputs(w_old, n_in, inserts):
    """First-layer weights for a network that gained inputs: copy the old rows and give
    the new inputs zero weight, so the warm-started network computes exactly what the old
    one did. `inserts` = [(row in the old input, number of new inputs)]."""
    rows, last = [], 0
    for at, k in inserts:
        rows += [w_old[last:at], jax.numpy.zeros((k, w_old.shape[1]))]
        last = at
    w = jax.numpy.concatenate(rows + [w_old[last:]])
    assert w.shape[0] == n_in, (w.shape, n_in)
    return w


def warm_start_teacher(old, fresh, sim):
    """Old long-range teacher -> visibility-limited one (3 new actor inputs; the critic
    also gets the true target)."""
    new = jax.tree.map(lambda x: x, old)
    if sim.vis_teacher and old["actor"][0]["w"].shape[0] != fresh["actor"][0]["w"].shape[0]:
        n_a = old["actor"][0]["w"].shape[0]
        new["actor"][0]["w"] = widen_inputs(old["actor"][0]["w"], fresh["actor"][0]["w"].shape[0], [(n_a, 3)])
        n_c = old["critic"][0]["w"].shape[0]
        new["critic"][0]["w"] = widen_inputs(old["critic"][0]["w"], fresh["critic"][0]["w"].shape[0],
                                             [(n_a, 3), (n_c, 4)])
    return new


def warm_start_student(old, old_sim, fresh):
    """Reuse a trained recurrent student in a student that sees more camera views: the
    CNN, GRU and head are copied; the new views' features start with zero weight, so it
    initially flies exactly like the old student."""
    new = jax.tree.map(lambda x: x, old)
    if "enc" not in old:                      # feed-forward student, same architecture: copy as is
        assert jax.tree.map(jax.numpy.shape, old) == jax.tree.map(jax.numpy.shape, fresh)
        return new
    w_old, w_new = old["enc"][0]["w"], fresh["enc"][0]["w"]
    if w_old.shape != w_new.shape:
        n_old = len(old_sim.views_hfov) + 4 * old_sim.wide_tiles
        flat = (w_new.shape[0] - w_old.shape[0]) // 4                      # 4 new tiles
        new["enc"][0]["w"] = widen_inputs(w_old, w_new.shape[0], [(n_old * flat, 4 * flat)])
    return new


def train_teacher(args, sim, out):
    overrides = {k: v for k, v in (("num_iters", args.iters), ("gamma", args.gamma)) if v is not None}
    cfg = ppo.Config(**overrides)
    init, train_iter = ppo.make_train(cfg, sim, args.dr)
    key, k = jax.random.split(jax.random.PRNGKey(args.seed))
    params, opt_state, states = init(k)
    if args.init:
        # the fresh optimizer state from init() already has the right shapes
        params = warm_start_teacher(load_ckpt(args.init)[0], params, sim)
    cur, log = Curriculum(sim, args.level0), Logger(cfg.num_iters, cfg.num_envs * cfg.rollout_len)
    for i in range(cfg.num_iters):
        key, k = jax.random.split(key)
        params, opt_state, states, m = train_iter(params, opt_state, states, k, cur.level)
        log(i, m, cur.level if cur.on else None)
        cur.update(m)
        if args.save_every and (i + 1) % args.save_every == 0:
            save_ckpt(out, params, sim, args, iteration=i, level=cur.level)
    return params, cur.level


def train_student(args, sim, out):
    teacher, teacher_sim = load_ckpt(args.teacher)
    overrides = {k: v for k, v in (("num_iters", args.iters), ("buffer_iters", args.buffer_iters),
                                   ("beta_iters", args.beta_iters), ("updates_per_iter", args.updates_per_iter),
                                   ("lr", args.lr))
                 if v is not None}
    cfg = dagger.Config(**overrides)
    init, train_iter = dagger.make_train(cfg, teacher, sim, args.dr, teacher_sim=teacher_sim)
    key, k = jax.random.split(jax.random.PRNGKey(args.seed))
    params, opt_state, states, frames, buffer = init(k)
    if args.init_student:
        params = warm_start_student(*load_ckpt(args.init_student), params)
    cur, log = Curriculum(sim, args.level0), Logger(cfg.num_iters, cfg.num_envs * cfg.rollout_len)
    for i in range(cfg.num_iters):
        key, k = jax.random.split(key)
        # DAgger: the teacher's share of control decays to 0. BC: the teacher always flies.
        beta = 1.0 if args.bc else (0.0 if cfg.beta_iters == 0 else max(0.0, 1.0 - i / cfg.beta_iters))
        # The student's curriculum is a fixed ramp to the full task over the first half
        # of training, so it can't stall below the success threshold.
        if cur.on and not sim.level_mix:
            cur.level = min(1.0, max(args.level0, i / (0.5 * cfg.num_iters)))
        params, opt_state, states, frames, buffer, m = train_iter(
            params, opt_state, states, frames, buffer, i, k, beta, cur.level)
        log(i, m, cur.level if cur.on else None)
        if args.save_every and (i + 1) % args.save_every == 0:
            save_ckpt(out, params, sim, args, iteration=i, level=cur.level)
            if args.keep_ckpts:
                save_ckpt(out.replace(".pkl", f"_it{i + 1}.pkl"), params, sim, args, iteration=i, level=cur.level)
    return params, cur.level


def train_finetune(args, sim, out):
    """RL fine-tuning of a trained student (its saved settings are reused)."""
    student, sim = load_ckpt(args.init_student)
    teacher, teacher_sim = load_ckpt(args.teacher)
    # a visibility-limited teacher needs the environment to track its belief state
    sim = sim._replace(vis_teacher=teacher_sim.vis_teacher)
    if args.episode_seconds:
        sim = sim._replace(max_steps=int(round(args.episode_seconds / sim.ctrl_dt)))
    if args.sag_floor is not None:
        sim = sim._replace(sag_floor=args.sag_floor)
    cfg = finetune.Config(**({"num_iters": args.iters} if args.iters else {}))
    init, train_iter = finetune.make_finetune(cfg, student, sim, teacher, teacher_sim, args.dr)
    key, k = jax.random.split(jax.random.PRNGKey(args.seed))
    nets, opts, states, frames_h = init(k)
    log = Logger(cfg.num_iters, cfg.num_envs * cfg.rollout_len)
    for i in range(cfg.num_iters):
        key, k = jax.random.split(key)
        nets, opts, states, frames_h, m = train_iter(nets, opts, states, frames_h, i, k)
        log(i, m)
        if args.save_every and (i + 1) % args.save_every == 0:
            save_ckpt(out, nets[0], sim, args, iteration=i)
    return nets[0], 1.0, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "student", "finetune"])
    ap.add_argument("--preset", choices=["short", "long"], default="short",
                    help="short = the task in the paper; long = 100-200 m targets, 40 m/s drone")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default="teacher.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dr", type=float, default=1.0, help="domain randomisation scale (0 = nominal drone)")
    ap.add_argument("--w-perception", type=float, default=None)
    ap.add_argument("--frame-stack", type=int, default=None)
    ap.add_argument("--bc", action="store_true", help="behaviour cloning instead of DAgger")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--buffer-iters", type=int, default=None, help="DAgger dataset size, in iterations")
    ap.add_argument("--init", default=None, help="warm-start the teacher from a checkpoint")
    ap.add_argument("--level0", type=float, default=0.0, help="starting curriculum level")
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint every N iterations (0 = off)")
    ap.add_argument("--level-mix", action="store_true", help="each episode samples its own curriculum level")
    ap.add_argument("--keep-ckpts", action="store_true", help="also keep a numbered copy of every checkpoint")
    ap.add_argument("--imu", action="store_true", help="student also gets the drone's IMU (gyro + accelerometer)")
    ap.add_argument("--recurrent", action="store_true", help="student with memory (GRU)")
    ap.add_argument("--hist-len", type=int, default=None, help="stick commands in the student's input")
    ap.add_argument("--episode-seconds", type=float, default=None, help="episode time limit")
    ap.add_argument("--sag-floor", type=float, default=None, help="battery keeps at least this fraction of thrust")
    ap.add_argument("--vis-teacher", action="store_true", help="teacher only knows the target while it is visible")
    ap.add_argument("--views", default=None, help="camera fields of view, e.g. 120,45,8")
    ap.add_argument("--wide-tiles", action="store_true", help="add the wide view at 2x resolution as 4 tiles")
    ap.add_argument("--init-student", default=None, help="warm-start the student from a checkpoint")
    ap.add_argument("--updates-per-iter", type=int, default=None, help="DAgger gradient steps per iteration")
    ap.add_argument("--lr", type=float, default=None, help="DAgger learning rate (cosine-decayed)")
    ap.add_argument("--beta-iters", type=int, default=None, help="DAgger: iterations until the teacher stops flying")
    args = ap.parse_args()

    sim = env.long_range_sim() if args.preset == "long" else env.Sim()
    if args.preset == "long" and args.gamma is None:
        args.gamma = 0.995            # 15 s episodes: look further ahead
    if args.level_mix:
        sim = sim._replace(level_mix=True)
    if args.imu:
        sim = sim._replace(imu=True)
    if args.recurrent:
        sim = sim._replace(recurrent=True)
    if args.hist_len:
        sim = sim._replace(hist_len=args.hist_len)
    if args.episode_seconds:
        sim = sim._replace(max_steps=int(round(args.episode_seconds / sim.ctrl_dt)))
    if args.sag_floor is not None:
        sim = sim._replace(sag_floor=args.sag_floor)
    if args.vis_teacher:
        sim = sim._replace(vis_teacher=True)
    if args.views:
        sim = sim._replace(views_hfov=tuple(float(v) for v in args.views.split(",")))
    if args.wide_tiles:
        sim = sim._replace(wide_tiles=True)
    sim = sim._replace(**{k: v for k, v in (("w_perception", args.w_perception),
                                            ("frame_stack", args.frame_stack)) if v is not None})
    out = args.out or f"{args.stage}.pkl"
    if args.stage == "finetune":
        params, level, sim = train_finetune(args, sim, out)
    else:
        params, level = (train_teacher if args.stage == "teacher" else train_student)(args, sim, out)
    save_ckpt(out, params, sim, args, level=level)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
