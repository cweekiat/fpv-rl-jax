"""Stage 2: distil the teacher into a camera-only *student* pilot with DAgger.

The student sees only what a real FPV pilot has:
    * the last few video frames (noisy, delayed, sometimes frozen)
    * its own recent stick inputs
and has to infer attitude, speed and target position from pixels.

DAgger (Dataset Aggregation): let the student fly, and at every state it visits ask
the teacher "what would you do here?". Add those labels to a growing dataset and
train the student to copy them. Because
the labels come from states the *student* reaches, it learns to recover from its
own mistakes. Plain imitation of teacher flights doesn't teach that.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from fpv import camera, env
from fpv.nets import cnn, crnn_encode, crnn_step, init_cnn, init_crnn
from fpv.ppo import episode_metrics, policy as teacher_policy


class Config(NamedTuple):
    num_envs: int = 512
    rollout_len: int = 32
    num_iters: int = 600
    buffer_iters: int = 12      # aggregated dataset = the last 12 iterations (~200k samples)
    updates_per_iter: int = 16  # minibatches drawn from the whole buffer each iteration
    minibatch: int = 2048
    lr: float = 3e-4
    beta_iters: int = 200       # teacher takes over less and less during this many iters
    teacher_gaze: bool = False  # if True the teacher also points the gaze while it flies
                                # (makes gaze trivial in its data; the student never learns to search)


def foveated(sim):
    return len(sim.views_hfov) > 1


def student_act(params, frames, states, sim, h=None):
    """Returns (sticks, gaze or None, image input, extra input, new memory or None).
    With foveation the student also outputs where its narrow views should look next,
    and knows where they look now. A recurrent student also carries memory `h`."""
    img = jax.vmap(lambda f, d: camera.pilot_view(f, d, sim))(frames, states.params.cam_delay)
    extra = states.act_buf[:, :sim.hist_len].reshape(states.act_buf.shape[0], -1)
    if sim.imu:
        extra = jnp.concatenate([extra, states.imu_buf.reshape(states.imu_buf.shape[0], -1)], axis=-1)
    if foveated(sim):
        extra = jnp.concatenate([extra, states.gaze], axis=-1)
    if sim.recurrent:
        h, out = crnn_step(params, h, crnn_encode(params, img, extra))
    else:
        out = cnn(params, img, extra)
    gaze = jnp.clip(out[:, env.ACT_DIM:], -1, 1) if foveated(sim) else None
    return out[:, :env.ACT_DIM], gaze, img, extra, h


def make_env_fns(sim, dr_scale):
    v_reset = jax.vmap(lambda k, lv: env.reset(k, sim, dr_scale, lv), in_axes=(0, None))
    v_step = jax.vmap(lambda k, s, a, lv: env.step(k, s, a, sim, dr_scale, lv), in_axes=(0, 0, 0, None))
    v_render = jax.vmap(lambda k, s: camera.render(k, s, sim))
    v_update = jax.vmap(camera.update_frames)

    def init_envs(key, n, level=1.0):
        k_r, k_c = jax.random.split(key)
        states = v_reset(jax.random.split(k_r, n), level)
        frames = jax.vmap(lambda f: camera.init_frames(f, sim))(
            v_render(jax.random.split(k_c, n), states))
        return states, frames

    def step(key, states, frames, action, level=1.0):
        k_step, k_cam, k_drop = jax.random.split(key, 3)
        n = action.shape[0]
        states, reward, done, info = v_step(jax.random.split(k_step, n), states, action, level)
        new = v_render(jax.random.split(k_cam, n), states)
        drop = jax.random.uniform(k_drop, (n,)) < states.params.drop_prob
        return states, v_update(frames, new, done, drop), reward, done, info

    return init_envs, step


def make_train(cfg: Config, teacher_params, sim: env.Sim = env.Sim(), dr_scale=1.0, teacher_sim=None):
    """DAgger. A feed-forward student learns from individual samples; a recurrent one
    from 32-step sequences, each stored with the memory it started with (truncated
    backpropagation through time, resetting the memory at episode boundaries)."""
    init_envs, env_step = make_env_fns(sim, dr_scale)
    tsim = teacher_sim or sim
    v_tobs = jax.vmap(lambda s: env.teacher_obs(s, tsim))
    tx = optax.adam(optax.cosine_decay_schedule(cfg.lr, cfg.num_iters * cfg.updates_per_iter, alpha=0.1))
    T, E = cfg.rollout_len, cfg.num_envs
    n = E * T
    n_views = len(sim.views_hfov) + 4 * sim.wide_tiles
    g = 2 if foveated(sim) else 0
    # student's non-image inputs: stick history (+ IMU samples) (+ current gaze)
    extra_dim = sim.hist_len * env.ACT_DIM + 6 * sim.imu_hist * sim.imu + 2 * foveated(sim)
    img_shape = (n_views, sim.img_h, sim.img_w, sim.frame_stack)
    # buffer slots: samples (feed-forward) or whole sequences (recurrent)
    slots, unit = (cfg.buffer_iters * E, (T,)) if sim.recurrent else (cfg.buffer_iters * n, ())

    def init(key):
        k_p, k_e = jax.random.split(key)
        states, frames = init_envs(k_e, E, 0.0 if sim.curriculum else 1.0)
        # The DAgger dataset. Images are stored as uint8 to keep it at a few hundred MB.
        buffer = (jnp.zeros((slots, *unit, *img_shape), jnp.uint8),
                  jnp.zeros((slots, *unit, extra_dim)),
                  jnp.zeros((slots, *unit, env.ACT_DIM + g)))
        if sim.recurrent:
            params = init_crnn(k_p, img_shape[1:], extra_dim, env.ACT_DIM + g, n_views=n_views, hidden=sim.gru_size)
            frames = (frames, jnp.zeros((E, sim.gru_size)))       # memory travels with the frames
            buffer = buffer + (jnp.zeros((slots, T), bool), jnp.zeros((slots, sim.gru_size)))
        else:
            params = init_cnn(k_p, img_shape[1:], extra_dim, env.ACT_DIM + g, n_views=n_views)
        return params, tx.init(params), states, frames, buffer

    def loss_fn(params, batch):
        img, hist, target = batch
        return jnp.mean((cnn(params, img.astype(jnp.float32) / 255.0, hist) - target) ** 2)

    def loss_seq(params, batch):
        img, extra, target, done, h0 = batch                     # (B, T, ...)
        b = img.shape[0]
        feat = crnn_encode(params, img.reshape(b * T, *img.shape[2:]).astype(jnp.float32) / 255.0,
                           extra.reshape(b * T, -1)).reshape(b, T, -1)

        def step(h, xs):
            f, d = xs
            h, out = crnn_step(params, h, f)
            return jnp.where(d[:, None], 0.0, h), out

        _, outs = jax.lax.scan(step, h0, (feat.swapaxes(0, 1), done.swapaxes(0, 1)))
        return jnp.mean((outs.swapaxes(0, 1) - target) ** 2)

    @partial(jax.jit, donate_argnums=(4,))
    def train_iter(params, opt_state, states, frames, buffer, it, key, beta, level=1.0):
        k_roll, k_upd = jax.random.split(key)
        frames, h_start = frames if sim.recurrent else (frames, None)

        def one_step(carry, key):
            states, frames, h = carry
            k_mix, k_env = jax.random.split(key)
            a_student, g_student, img, extra, h_new = student_act(params, frames, states, sim, h)
            a_teacher = jnp.clip(teacher_policy(teacher_params, v_tobs(states))[0], -1, 1)
            # Early on the teacher flies some of the time (beta), so the student isn't
            # learning only from crash states.
            use_teacher = jax.random.uniform(k_mix, (E, 1)) < beta
            action = jnp.where(use_teacher, a_teacher, a_student)
            label = a_teacher
            if foveated(sim):
                # Gaze label: where the target really is in the wide view.
                g_label = jax.vmap(lambda s: camera.target_gaze(s, sim))(states)
                gaze = jnp.where(use_teacher, g_label, g_student) if cfg.teacher_gaze else g_student
                states = states._replace(gaze=gaze)
                label = jnp.concatenate([a_teacher, g_label], axis=-1)
            gaze_err = jnp.mean(jnp.abs(g_student - g_label)) if foveated(sim) else jnp.zeros(())
            states, frames, _, done, info = env_step(k_env, states, frames, action, level)
            if sim.recurrent:
                h_new = jnp.where(done[:, None], 0.0, h_new)       # fresh memory for a new episode
            img_u8 = jnp.round(img * 255.0).astype(jnp.uint8)
            return (states, frames, h_new), ((img_u8, extra, label), done, info, gaze_err)

        (states, frames, h_end), (data, done, info, gaze_err) = jax.lax.scan(
            one_step, (states, frames, h_start), jax.random.split(k_roll, T))

        # Aggregate: overwrite the oldest slot of the ring buffer with this iteration's data.
        if sim.recurrent:
            seqs = jax.tree.map(lambda x: x.swapaxes(0, 1), data) + (done.swapaxes(0, 1), h_start)
            start, filled, mb, lossf = (it % cfg.buffer_iters) * E, jnp.minimum(it + 1, cfg.buffer_iters) * E, \
                cfg.minibatch // T, loss_seq
            new = seqs
        else:
            new = jax.tree.map(lambda x: x.reshape(n, *x.shape[2:]), data)
            start, filled, mb, lossf = (it % cfg.buffer_iters) * n, jnp.minimum(it + 1, cfg.buffer_iters) * n, \
                cfg.minibatch, loss_fn
        buffer = jax.tree.map(lambda b, x: jax.lax.dynamic_update_slice_in_dim(b, x, start, 0), buffer, new)

        def update(carry, key):
            params, opt_state = carry
            idx = jax.random.randint(key, (mb,), 0, filled)
            loss, grads = jax.value_and_grad(lossf)(params, jax.tree.map(lambda b: b[idx], buffer))
            updates, opt_state = tx.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), loss

        (params, opt_state), losses = jax.lax.scan(update, (params, opt_state),
                                                   jax.random.split(k_upd, cfg.updates_per_iter))
        metrics = episode_metrics(done, info)
        metrics["imitation_mse"] = losses.mean()
        metrics["gaze_err"] = gaze_err.mean()
        frames = (frames, h_end) if sim.recurrent else frames
        return params, opt_state, states, frames, buffer, metrics

    return init, train_iter
