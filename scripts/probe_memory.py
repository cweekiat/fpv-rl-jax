"""What has the recurrent student's memory learned? Roll the student out, record its GRU
hidden state at every step, and fit linear (ridge) readouts from the hidden state to
quantities it was never told: attitude, velocity, altitude, target geometry, and the
hidden drone parameters. Reported as R^2 on held-out episodes.

    uv run python scripts/probe_memory.py ckpt/lr_student_gru.pkl
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpv import env  # noqa: E402
from fpv.dagger import make_env_fns, student_act  # noqa: E402
from fpv.nets import conv_features  # noqa: E402
from train import load_ckpt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/lr_student_gru.pkl"
params, sim = load_ckpt(path)
sim = sim._replace(level_mix=False)
# memoryless students are probed on their penultimate features, as a control:
# whatever they encode comes from the current inputs alone (frames + stick history)
N, T, EVERY = 256, sim.max_steps, 5


def rollout(key):
    init_envs, env_step = make_env_fns(sim, 1.0)
    k0, k1 = jax.random.split(key)
    states, frames = init_envs(k0, N)

    def one(carry, k):
        states, frames, h, active = carry
        a, g, img, extra, h_new = student_act(params, frames, states, sim, h if sim.recurrent else None)
        if sim.recurrent:
            h = feat = h_new
        else:
            x = jnp.concatenate([conv_features(params["convs"], img), extra], axis=-1)
            feat = jax.nn.relu(x @ params["head"][0]["w"] + params["head"][0]["b"])
        R = jax.vmap(env.quat_to_rotmat)(states.quat)
        rel = states.target - states.pos
        feats = {
            "h": feat,
            "gravity_dir_body": R[:, 2, :],                                   # attitude
            "velocity_body": jnp.einsum("nji,nj->ni", R, states.vel),
            "speed": jnp.linalg.norm(states.vel, axis=-1, keepdims=True),
            "altitude": states.pos[:, 2:3],
            "body_rates": states.omega,
            "log_target_distance": jnp.log1p(jnp.linalg.norm(rel, axis=-1, keepdims=True)),
            "target_dir_body": jnp.einsum("nji,nj->ni", R, rel / jnp.linalg.norm(rel, axis=-1, keepdims=True)),
            "thrust_to_weight": states.params.twr[:, None],
            "latency_steps": (states.params.act_delay + states.params.cam_delay)[:, None].astype(jnp.float32),
            "wind_world": states.params.wind_mean[:, :2] + states.gust[:, :2],
            "active": active[:, None].astype(jnp.float32),
        }
        states = states._replace(gaze=g)
        states, frames, _, done, _ = env_step(k, states, frames, a)
        if sim.recurrent:
            h = jnp.where(done[:, None], 0.0, h)
        return (states, frames, h, active & ~done), feats

    h0 = jnp.zeros((N, sim.gru_size)) if sim.recurrent else jnp.zeros((N, 1))
    _, feats = jax.lax.scan(one, (states, frames, h0, jnp.ones(N, bool)),
                            jax.random.split(k1, T))
    return feats


feats = jax.device_get(jax.jit(rollout)(jax.random.PRNGKey(11)))
# keep active steps (first episode only), subsampled in time, split train/test by episode
mask = feats["active"][::EVERY, :, 0] > 0
ep = np.broadcast_to(np.arange(N), mask.shape)[mask]
X = feats["h"][::EVERY][mask]
train, test = ep % 4 != 0, ep % 4 == 0


def ridge_r2(y, lam=1.0):
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = (X - mu) / sd
    A = np.c_[Xs, np.ones(len(Xs))]
    w = np.linalg.solve(A[train].T @ A[train] + lam * np.eye(A.shape[1]), A[train].T @ y[train])
    pred = A[test] @ w
    ss_res = ((y[test] - pred) ** 2).sum(0)
    ss_tot = ((y[test] - y[test].mean(0)) ** 2).sum(0) + 1e-9
    return float(np.mean(1 - ss_res / ss_tot))


out = {"checkpoint": path, "samples": int(mask.sum()), "features": "gru" if sim.recurrent else "penultimate"}
for k, v in feats.items():
    if k in ("h", "active"):
        continue
    out[k] = ridge_r2(v[::EVERY][mask])
    print(f"{k:22s} R^2 = {out[k]:.2f}")
# chance level: the same probe on shuffled targets
rng = np.random.default_rng(0)
out["shuffled_gravity_dir_body"] = ridge_r2(feats["gravity_dir_body"][::EVERY][mask][rng.permutation(int(mask.sum()))])
print(f"{'shuffled (chance)':22s} R^2 = {out['shuffled_gravity_dir_body']:.2f}")
os.makedirs("results", exist_ok=True)
name = os.path.basename(path).removesuffix(".pkl")
with open(f"results/probe_{name}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"saved results/probe_{name}.json")
