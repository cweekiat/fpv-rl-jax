"""Stage 3 (optional): fine-tune a DAgger student with reinforcement learning.

Imitation can only copy the teacher. A camera-only pilot sometimes needs to do
something the teacher never had to: search for a target it can't see, and recover
from its own mistakes. Here the recurrent student itself is trained with PPO on its
own camera inputs:

  * actor   = the student (CNN + GRU), with a Gaussian over sticks and gaze
  * critic  = a privileged MLP on the true state (asymmetric actor-critic), warm-started
              from the teacher's critic
  * anchor  = an imitation term pulling the student toward the teacher's labels, decayed
              over training, so RL refines the policy instead of re-learning it

Sequences of 32 steps are stored with the GRU state they started from, as in DAgger.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from fpv import camera, env
from fpv.dagger import foveated, make_env_fns, student_act
from fpv.nets import cnn, crnn_encode, crnn_step, mlp
from fpv.ppo import episode_metrics, log_prob, policy as teacher_policy


class Config(NamedTuple):
    num_envs: int = 256
    rollout_len: int = 32
    num_iters: int = 500
    epochs: int = 2
    num_minibatches: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.1          # conservative: refine, don't rewrite
    lr: float = 2e-5
    critic_lr: float = 3e-4
    anchor_start: float = 1.0      # weight of the imitation anchor, decayed linearly...
    anchor_end: float = 0.1        # ...to this
    log_std_init: float = -2.0


def make_finetune(cfg: Config, student_params, sim, teacher_params, teacher_sim, dr_scale=1.0):
    assert foveated(sim), "fine-tuning is written for the foveated (gaze) student"
    init_envs, env_step = make_env_fns(sim, dr_scale)
    v_tobs = jax.vmap(lambda s: env.teacher_obs(s, teacher_sim))
    v_cobs = jax.vmap(lambda s: env.critic_obs(s, teacher_sim))
    T, E = cfg.rollout_len, cfg.num_envs
    n_act = env.ACT_DIM + 2
    tx_pi = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    tx_v = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.critic_lr))

    def init(key):
        states, frames = init_envs(key, E, 1.0)
        actor = dict(student_params, log_std=jnp.full(n_act, cfg.log_std_init))
        critic = teacher_params["critic"]
        h = jnp.zeros((E, sim.gru_size if sim.recurrent else 1))      # memoryless students carry a dummy
        return (actor, critic), (tx_pi.init(actor), tx_v.init(critic)), states, (frames, h)

    def value(critic, cobs):
        return mlp(critic, cobs)[..., 0]

    def rollout(actor, critic, states, frames, h, key):
        def one(carry, key):
            states, frames, h = carry
            k_a, k_env = jax.random.split(key)
            a_mean, g_mean, img, extra, h_new = student_act(actor, frames, states, sim, h if sim.recurrent else None)
            h_new = h if h_new is None else h_new
            mean = jnp.concatenate([a_mean, g_mean], -1)
            act = mean + jnp.exp(actor["log_std"]) * jax.random.normal(k_a, mean.shape)
            logp = log_prob(mean, actor["log_std"], act)
            cobs = v_cobs(states)
            label = jnp.concatenate([jnp.clip(teacher_policy(teacher_params, v_tobs(states))[0], -1, 1),
                                     jax.vmap(lambda s: camera.target_gaze(s, sim))(states)], -1)
            states = states._replace(gaze=jnp.clip(act[:, env.ACT_DIM:], -1, 1))
            states, frames, reward, done, info = env_step(k_env, states, frames, act[:, :env.ACT_DIM])
            h_new = jnp.where(done[:, None], 0.0, h_new)
            img_u8 = jnp.round(img * 255.0).astype(jnp.uint8)
            return (states, frames, h_new), (img_u8, extra, act, logp, value(critic, cobs), cobs, reward, done,
                                             label, info)

        (states, frames, h_end), traj = jax.lax.scan(one, (states, frames, h), jax.random.split(key, T))
        return states, frames, h_end, traj

    def gae(rewards, values, dones, last_value):
        def back(carry, x):
            adv, next_v = carry
            r, v, d = x
            nonterm = 1.0 - d
            delta = r + cfg.gamma * next_v * nonterm - v
            adv = delta + cfg.gamma * cfg.gae_lambda * nonterm * adv
            return (adv, v), adv

        _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True)
        return adv, adv + values

    def actor_loss(actor, batch, anchor_w):
        img, extra, act, old_logp, adv, done, label, h0 = batch      # (B, T, ...)
        b = img.shape[0]
        img_f = img.reshape(b * T, *img.shape[2:]).astype(jnp.float32) / 255.0
        if sim.recurrent:
            feat = crnn_encode(actor, img_f, extra.reshape(b * T, -1)).reshape(b, T, -1)

            def step(h, xs):
                f, d = xs
                h, out = crnn_step(actor, h, f)
                return jnp.where(d[:, None], 0.0, h), out

            _, out = jax.lax.scan(step, h0, (feat.swapaxes(0, 1), done.swapaxes(0, 1)))
            mean = out.swapaxes(0, 1)
        else:
            mean = cnn(actor, img_f, extra.reshape(b * T, -1)).reshape(b, T, -1)
        mean = mean.at[..., env.ACT_DIM:].set(jnp.clip(mean[..., env.ACT_DIM:], -1, 1))
        ratio = jnp.exp(log_prob(mean, actor["log_std"], act) - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        anchor = jnp.mean((mean - label) ** 2)
        return pg + anchor_w * anchor, (pg, anchor)

    def critic_loss(critic, cobs, ret):
        return 0.5 * jnp.mean((value(critic, cobs) - ret) ** 2)

    @jax.jit
    def train_iter(nets, opts, states, frames_h, it, key):
        (actor, critic), (opt_pi, opt_v) = nets, opts
        frames, h0 = frames_h
        k_roll, k_upd = jax.random.split(key)
        states, frames, h_end, traj = rollout(actor, critic, states, frames, h0, k_roll)
        img, extra, act, logp, values, cobs, rewards, dones, labels, info = traj
        adv, ret = gae(rewards, values, dones, value(critic, v_cobs(states)))
        seq = lambda x: x.swapaxes(0, 1)                                  # (T, E, ...) -> (E, T, ...)
        data = (seq(img), seq(extra), seq(act), seq(logp), seq(adv), seq(dones), seq(labels), h0)
        anchor_w = cfg.anchor_start + (cfg.anchor_end - cfg.anchor_start) * it / cfg.num_iters

        def epoch(carry, key):
            def minibatch(carry, idx):
                actor, critic, opt_pi, opt_v = carry
                batch = jax.tree.map(lambda x: x[idx], data)
                (loss, aux), g_pi = jax.value_and_grad(actor_loss, has_aux=True)(actor, batch, anchor_w)
                up, opt_pi = tx_pi.update(g_pi, opt_pi)
                actor = optax.apply_updates(actor, up)
                vl, g_v = jax.value_and_grad(critic_loss)(critic, seq(cobs)[idx].reshape(-1, cobs.shape[-1]),
                                                          seq(ret)[idx].reshape(-1))
                up, opt_v = tx_v.update(g_v, opt_v)
                critic = optax.apply_updates(critic, up)
                return (actor, critic, opt_pi, opt_v), (aux[0], aux[1], vl)

            idx = jax.random.permutation(key, E).reshape(cfg.num_minibatches, -1)
            return jax.lax.scan(minibatch, carry, idx)

        (actor, critic, opt_pi, opt_v), (pg, anchor, vl) = jax.lax.scan(
            epoch, (actor, critic, opt_pi, opt_v), jax.random.split(k_upd, cfg.epochs))
        metrics = episode_metrics(dones, info)
        metrics.update(reward=rewards.mean(), imitation_mse=anchor.mean(), v_loss=vl.mean())
        return (actor, critic), (opt_pi, opt_v), states, (frames, h_end), metrics

    return init, train_iter
