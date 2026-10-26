"""A hand-designed visual-servoing pilot: the non-learned baseline.

It sees exactly what the camera-only student sees (the delayed, noisy views and its
own stick history) and flies with classical rules:
  * attitude from the horizon: fit a line to the sky/ground boundary in the wide view
  * target from the image: the darkest blob in the narrowest view that contains it
  * pursuit guidance: yaw and bank toward the target, pitch forward to a cruise angle,
    throttle compensating for tilt and for the target's elevation
  * memory of the last sighting: steer the gaze there, and search when the target is lost
Gains are tuned by a small grid search (scripts/visual_servo.py).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from fpv import camera, env


class Gains(NamedTuple):
    cruise_pitch: float = 0.8     # rad nose-down when the target is level and far
    k_pitch: float = 4.0          # rate per rad of pitch error
    k_roll: float = 4.0
    bank: float = 1.0             # desired bank per rad of target azimuth
    k_yaw: float = 2.0
    k_climb: float = 1.5          # extra thrust per rad of target elevation
    twr_guess: float = 5.5        # assumed thrust-to-weight
    tilt_guess: float = 0.785     # assumed camera uptilt (45 deg)
    detect: float = 0.12          # how much darker than the image median counts as target
    k_los: float = 2.0            # thrust damping per rad/s of line-of-sight elevation rate
    calibrated: bool = True       # know the true camera uptilt and thrust-to-weight (generous:
                                  # the learned students are never given these)


def horizon(img, sim):
    """(roll, camera elevation) from the sky/ground boundary of the wide view.
    Sky is brighter than ground; per column, count sky pixels from the top."""
    h, w = img.shape
    thr = 0.5 * (jnp.percentile(img, 90) + jnp.percentile(img, 10))
    sky = (img > thr).astype(jnp.float32)
    n = sky.sum(0)                                             # sky pixels per column
    valid = ((n > 0.5) & (n < h - 0.5)).astype(jnp.float32)
    u = (jnp.arange(w) + 0.5) / w * 2 - 1
    v = 2 * n / h - 1                                          # horizon row, -1 top .. 1 bottom
    sw = valid.sum() + 1e-6
    um, vm = (valid * u).sum() / sw, (valid * v).sum() / sw
    b = (valid * (u - um) * (v - vm)).sum() / ((valid * (u - um) ** 2).sum() + 1e-6)
    a = vm - b * um
    tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[0]) / 2)
    tan_v = tan_h * h / w
    roll = jnp.arctan(b * tan_v / tan_h)
    elev = jnp.arctan(a * tan_v)                               # camera axis above the horizon
    # no horizon in view: mostly sky -> looking up; mostly ground -> looking down
    frac_sky = sky.mean()
    elev = jnp.where(sw < 4, jnp.where(frac_sky > 0.5, 0.9, -0.9), elev)
    roll = jnp.where(sw < 4, 0.0, roll)
    return roll, elev


def find_target(views, gaze, sim, g: Gains):
    """Direction (camera frame) of the darkest blob, searching the narrow views first."""
    found, direction = jnp.array(False), jnp.array([1.0, 0.0, 0.0])
    for k in reversed(range(len(sim.views_hfov))):
        img = views[k]
        idx = jnp.argmin(img)
        darkness = jnp.median(img) - img.reshape(-1)[idx]
        r, c = idx // sim.img_w, idx % sim.img_w
        tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[k]) / 2)
        tan_v = tan_h * sim.img_h / sim.img_w
        u = (c + 0.5) / sim.img_w * 2 - 1
        v = (r + 0.5) / sim.img_h * 2 - 1
        d = jnp.array([1.0, -u * tan_h, -v * tan_v])
        if k > 0:   # narrow views are rotated toward the gaze point of the wide view
            gd = camera.gaze_direction(gaze, sim)
            yaw, pitch = jnp.arctan2(gd[1], gd[0]), jnp.arcsin(jnp.clip(gd[2], -1, 1))
            cy, sy = jnp.cos(yaw), jnp.sin(yaw)
            rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            d = rz @ camera._rot_y_up(pitch) @ d
        hit = (darkness > g.detect) & ~found
        direction = jnp.where(hit, d / jnp.linalg.norm(d), direction)
        found = found | hit
    return found, direction


def dir_to_gaze(d, sim):
    tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[0]) / 2)
    tan_v = tan_h * sim.img_h / sim.img_w
    x = jnp.maximum(d[0], 1e-3)
    return jnp.clip(jnp.array([-d[1] / x / tan_h, -d[2] / x / tan_v]), -1, 1)


class ServoPilot:
    """Stateful pilot for evaluate.run: memory = [last target direction (3), steps since seen,
    last target elevation]."""

    def __init__(self, sim, gains=Gains()):
        self.sim, self.g = sim, gains

    def init(self, n):
        return jnp.tile(jnp.array([1.0, 0.0, -0.2, 99.0, 0.0]), (n, 1))

    def act(self, states, frames, mem):
        sim, g = self.sim, self.g

        def one(frame_buf, cam_delay, gaze, m, tilt_true, twr_true):
            views = camera.pilot_view(frame_buf, cam_delay, sim)[..., 0]     # newest received frame per view
            roll, elev = horizon(views[0], sim)
            found, d_cam = find_target(views, gaze, sim, g)
            d_cam = jnp.where(found, d_cam, m[:3])
            since = jnp.where(found, 0.0, m[3] + 1)
            # target bearing relative to the body: azimuth (left +), elevation above the horizon
            az = jnp.arctan2(d_cam[1], d_cam[0])
            el = jnp.arcsin(jnp.clip(d_cam[2], -1, 1)) + elev
            tilt_cam = tilt_true if g.calibrated else g.tilt_guess
            twr = twr_true if g.calibrated else g.twr_guess
            pitch_up = elev - tilt_cam                             # body pitch, nose-up positive
            el_rate = jnp.where(found & (m[3] < 1), (el - m[4]) / sim.ctrl_dt, 0.0)
            lost = since > 25
            # pursuit: pitch forward to cruise (less when the target is high), bank into the turn
            # never steeper than the cruise angle, so the thrust can still hold altitude
            want_down = jnp.where(lost, 0.3, jnp.clip(g.cruise_pitch - 1.5 * el, 0.0, g.cruise_pitch))
            want_roll_right = jnp.clip(-g.bank * az, -0.8, 0.8)
            w_pitch = g.k_pitch * (want_down - (-pitch_up))       # +y rate = nose down
            # the horizon fit gives roll positive for left-wing-down; +x rate rolls right
            w_roll = g.k_roll * (want_roll_right - (-roll))
            w_yaw = jnp.where(lost, 0.6, g.k_yaw * az)            # lost: turn slowly to search
            tilt = jnp.cos(roll) * jnp.cos(pitch_up)
            thrust = (1.0 + g.k_climb * jnp.clip(el, -0.5, 0.5) + g.k_los * jnp.clip(el_rate, -0.3, 0.3)) \
                / jnp.maximum(tilt, 0.3)
            throttle = 2.0 * thrust / twr - 1.0
            action = jnp.clip(jnp.array([throttle, w_roll / sim.max_rate, w_pitch / sim.max_rate,
                                         w_yaw / sim.max_rate]), -1, 1)
            new_gaze = jnp.where(found | ~lost, dir_to_gaze(d_cam, sim), jnp.array([0.0, 0.3]))
            return action, new_gaze, jnp.concatenate([d_cam, since[None], el[None]]), roll, elev

        a, gz, m, _, _ = jax.vmap(one)(frames, states.params.cam_delay, states.gaze, mem,
                                       states.params.cam_tilt, states.params.twr)
        return a, gz, m
