"""Stage 1: train the privileged *teacher* with PPO, in pure JAX + Optax.

The teacher sees the true state (it can only exist in simulation). It does have to
cope with domain randomisation and all the real-world effects, because it can't
see the randomised parameters.

Asymmetric actor-critic: the critic also gets the hidden parameters (wind, drone
properties, time left). This makes value estimates much better, and the critic is
thrown away after training, so it's allowed to "cheat".
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from fpv import env
from fpv.nets import init_mlp, mlp


class Config(NamedTuple):
    num_envs: int = 1024
    rollout_len: int = 128
    num_iters: int = 600
    epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 256


def init_params(key, obs_dim, critic_dim, cfg: Config):
    k_pi, k_v = jax.random.split(key)
    h = cfg.hidden
    return {
        "actor": init_mlp(k_pi, [obs_dim, h, h, env.ACT_DIM], 0.01),
        "critic": init_mlp(k_v, [critic_dim, h, h, 1], 1.0),
        "log_std": jnp.full(env.ACT_DIM, -0.5),
    }


def policy(params, obs):
    return mlp(params["actor"], obs), params["log_std"]


def value(params, cobs):
    return mlp(params["critic"], cobs)[..., 0]


def log_prob(mean, log_std, a):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2
                   - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


class Transition(NamedTuple):
    obs: jax.Array
    cobs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


def make_train(cfg: Config, sim: env.Sim = env.Sim(), dr_scale=1.0):
    v_reset = jax.vmap(lambda k, lv: env.reset(k, sim, dr_scale, lv), in_axes=(0, None))
    v_step = jax.vmap(lambda k, s, a, lv: env.step(k, s, a, sim, dr_scale, lv), in_axes=(0, 0, 0, None))
    v_obs = jax.vmap(lambda s: (env.teacher_obs(s, sim), env.critic_obs(s, sim)))

    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(optax.linear_schedule(cfg.lr, cfg.lr * 0.1,
                                                      cfg.num_iters * cfg.epochs * cfg.num_minibatches),
                                eps=1e-5))

    def init(key):
        k_p, k_e = jax.random.split(key)
        states = v_reset(jax.random.split(k_e, cfg.num_envs), 0.0 if sim.curriculum else 1.0)
        obs, cobs = v_obs(states)
        params = init_params(k_p, obs.shape[-1], cobs.shape[-1], cfg)
        return params, tx.init(params), states

    def rollout(params, states, key, level):
        def one_step(states, key):
            k_act, k_env = jax.random.split(key)
            obs, cobs = v_obs(states)
            mean, log_std = policy(params, obs)
            action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
            states, reward, done, info = v_step(jax.random.split(k_env, cfg.num_envs), states, action, level)
            return states, Transition(obs, cobs, action, log_prob(mean, log_std, action),
                                      value(params, cobs), reward, done, info)

        return jax.lax.scan(one_step, states, jax.random.split(key, cfg.rollout_len))

    def gae(traj, last_value):
        def back(carry, t):
            adv, next_v = carry
            nonterm = 1.0 - t.done
            delta = t.reward + cfg.gamma * next_v * nonterm - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * nonterm * adv
            return (adv, t.value), adv

        _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        return adv, adv + traj.value

    def loss_fn(params, batch):
        obs, cobs, action, old_logp, adv, ret = batch
        mean, log_std = policy(params, obs)
        ratio = jnp.exp(log_prob(mean, log_std, action) - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.minimum(ratio * adv,
                               jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_loss = 0.5 * ((value(params, cobs) - ret) ** 2).mean()
        return pg_loss + cfg.vf_coef * v_loss, (pg_loss, v_loss)

    def update(params, opt_state, traj, adv, ret, key):
        n = cfg.num_envs * cfg.rollout_len
        flat = jax.tree.map(lambda x: x.reshape(n, *x.shape[2:]),
                            (traj.obs, traj.cobs, traj.action, traj.logp, adv, ret))

        def epoch(carry, key):
            perm = jax.random.permutation(key, n)
            mbs = jax.tree.map(lambda x: x[perm].reshape(cfg.num_minibatches, -1, *x.shape[1:]), flat)

            def minibatch(carry, mb):
                params, opt_state = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = tx.update(grads, opt_state)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(minibatch, carry, mbs)

        (params, opt_state), losses = jax.lax.scan(epoch, (params, opt_state),
                                                   jax.random.split(key, cfg.epochs))
        return params, opt_state, jax.tree.map(jnp.mean, losses)

    @jax.jit
    def train_iter(params, opt_state, states, key, level=1.0):
        k_roll, k_upd = jax.random.split(key)
        states, traj = rollout(params, states, k_roll, level)
        adv, ret = gae(traj, value(params, v_obs(states)[1]))
        params, opt_state, (pg, vl) = update(params, opt_state, traj, adv, ret, k_upd)
        metrics = episode_metrics(traj.done, traj.info)
        metrics.update(reward=traj.reward.mean(), v_loss=vl)
        return params, opt_state, states, metrics

    return init, train_iter


def episode_metrics(done, info):
    n = jnp.maximum(done.sum(), 1)
    succ = info["success"]
    return {
        "success": succ.sum() / n,
        "crash": info["crashed"].sum() / n,
        "timeout": info["timeout"].sum() / n,
        "time_to_target": jnp.where(succ, info["time"], 0.0).sum() / jnp.maximum(succ.sum(), 1),
    }
