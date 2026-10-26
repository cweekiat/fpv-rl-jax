"""FPV quadrotor environment in pure JAX: race to a target as fast as possible.

The agent flies like an FPV pilot in acro mode:
    action[0]   throttle        -> collective thrust
    action[1:4] roll/pitch/yaw  -> desired body angular rates

Every episode samples a different drone and world (domain randomisation), and
the simulator includes real-world effects: actuator latency, motor lag, limited
torque and saturation, battery sag, wind and gusts, ground effect, and thrust
and vibration noise.

The simulator knows the full state. A real pilot doesn't, so there are two
observation functions:
    teacher_obs  - privileged, noise-free state (for training in simulation only)
    camera.py    - what the pilot actually sees: a noisy, delayed FPV video feed
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Sim(NamedTuple):
    """Fixed simulator settings (the same for every episode)."""
    dt: float = 0.01            # physics step (100 Hz)
    substeps: int = 2           # control at 50 Hz
    gravity: float = 9.81
    max_rate: float = 8.0       # rad/s at full stick (~460 deg/s, "pilot rates")
    max_steps: int = 250        # 5 s episodes at 50 Hz
    goal_radius: float = 0.5    # m, counts as "reached the target"
    # task geometry
    target_dist: tuple = (4.0, 12.0)   # horizontal spawn distance to target (m)
    target_alt: tuple = (1.0, 4.0)     # target altitude (m)
    ceiling: float = 15.0              # crash above this altitude (m)
    max_dist: float = 25.0             # crash if further than this from the target (m)
    max_act_delay: int = 2      # control steps (up to 40 ms)
    max_cam_delay: int = 2      # control steps (up to 40 ms of video latency)
    # camera
    img_h: int = 24
    img_w: int = 32
    hfov_deg: float = 120.0     # typical wide FPV lens
    frame_stack: int = 3
    target_radius: float = 0.3
    hist_len: int = 3           # past stick commands given to the policy
    imu: bool = False           # give the student the drone's IMU (gyro + accelerometer)
    imu_hist: int = 3           # number of recent IMU samples it sees
    recurrent: bool = False     # student has memory (a GRU) instead of only frame stacking
    gru_size: int = 256
    vis_teacher: bool = False   # teacher only knows the target while the student could see it
    vis_range: float = 150.0    # ...i.e. inside the wide view and closer than this (m)
    wide_tiles: bool = False    # also render the wide view at 2x resolution, as 4 extra tiles
    sag_floor: float = 0.0      # battery never sags below this fraction of full thrust
                                # (0 = plain linear sag, fine for <=15 s; 0.7 for long flights)
    w_perception: float = 0.05  # reward weight for keeping the target in view
    time_penalty: float = 0.01  # per-step cost: be quick
    target_bearing_deg: float = 50.0   # target spawns within this angle of the heading
    # --- long-range variants. The defaults reproduce the short-range task in the paper.
    curriculum: bool = False           # if True, `level` in [0, 1] interpolates the task
    easy_dist: tuple = (4.0, 12.0)     # ...from these level-0 ranges to the ones above
    easy_alt: tuple = (1.0, 4.0)
    easy_bearing_deg: float = 50.0
    level_mix: bool = False            # each episode samples its own level in [0, level]
    progress_norm: bool = False        # progress reward relative to the start distance
    obs_v2: bool = False               # direction + log-distance + altitude observation
    dr_twr: tuple = (2.5, 5.0)
    dr_drag_lin: tuple = (0.1, 0.5)
    dr_drag_quad: tuple = (0.0, 0.05)
    dr_cam_tilt_deg: tuple = (20.0, 40.0)
    views_hfov: tuple = (120.0,)       # several fields of view of one camera = foveation
    area_target: bool = False          # physically correct sub-pixel target, no ~1 px glow
    fog_dist: float = 40.0             # m, haze distance
    terrain_layers: bool = False       # coarse ground texture + texture mip-mapping

    @property
    def ctrl_dt(self):
        return self.dt * self.substeps


class DRParams(NamedTuple):
    """Per-episode randomised drone and world. The policy never sees these directly."""
    twr: jax.Array          # thrust-to-weight ratio
    rate_tau: jax.Array     # flight-controller rate tracking time constant (s)
    thrust_tau: jax.Array   # motor spin-up time constant (s)
    drag_lin: jax.Array     # linear drag (1/s)
    drag_quad: jax.Array    # quadratic drag (1/m)
    ang_acc: jax.Array      # max roll/pitch angular acceleration (rad/s^2) = torque limit
    yaw_auth: jax.Array     # yaw authority relative to roll/pitch (props give weak yaw)
    sag_rate: jax.Array     # battery sag: fraction of max thrust lost per second
    wind_mean: jax.Array    # (3,) steady wind (m/s)
    gust_sigma: jax.Array   # gust strength (m/s)
    act_delay: jax.Array    # int, control steps of radio + FC latency
    cam_delay: jax.Array    # int, control steps of video latency
    cam_tilt: jax.Array     # camera uptilt (rad)
    tile: jax.Array         # ground texture scale (m), so visual scale is ambiguous
    gain: jax.Array         # camera exposure
    offset: jax.Array       # camera black level
    pix_noise: jax.Array    # per-pixel sensor noise std
    drop_prob: jax.Array    # probability a video frame freezes (link breakup)
    # IMU (MEMS gyro + accelerometer), randomised per episode
    imu_scale: jax.Array    # overall noise level relative to nominal (1 = typical)
    gyro_bias: jax.Array    # (3,) rad/s, constant over the episode
    gyro_noise: jax.Array   # rad/s white noise per sample
    gyro_vib: jax.Array     # rad/s extra noise at full throttle (motor vibration)
    accel_bias: jax.Array   # (3,) m/s^2
    accel_noise: jax.Array  # m/s^2
    accel_vib: jax.Array    # m/s^2 at full throttle


class DroneState(NamedTuple):
    pos: jax.Array        # (3,) world, z up
    vel: jax.Array        # (3,)
    quat: jax.Array       # (4,) body->world (w, x, y, z); body x = forward, z = up
    omega: jax.Array      # (3,) body rates
    thrust: jax.Array     # () collective thrust per unit mass (m/s^2)
    act_buf: jax.Array    # (max_act_delay+1, 4) recent stick commands, newest first
    gust: jax.Array       # (3,) current gust (Ornstein-Uhlenbeck process)
    target: jax.Array     # (3,)
    prev_dist: jax.Array  # ()
    d0: jax.Array         # () start distance
    level: jax.Array      # () curriculum level the episode was spawned at
    gaze: jax.Array       # (2,) where the narrow camera views point, in wide-view image coords
    imu_buf: jax.Array    # (imu_hist, 6) recent IMU samples (gyro, accel), newest first
    belief: jax.Array     # (3,) where a visibility-limited pilot thinks the target is (world)
    seen: jax.Array       # () 1 once the target has been in view
    since_seen: jax.Array  # () seconds since it was last in view
    t: jax.Array          # () control step counter
    params: DRParams


# ---------------------------------------------------------------- math helpers
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rotmat(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_axis_body(tilt):
    """Optical axis in the body frame: forward, pitched up by `tilt`."""
    return jnp.array([jnp.cos(tilt), 0.0, jnp.sin(tilt)])


# ---------------------------------------------------------------- randomisation
def sample_params(key, scale=1.0, sim: "Sim" = None) -> DRParams:
    """scale=0 gives the nominal drone; scale=1 gives the full randomisation range."""
    sim = sim or Sim()
    ks = jax.random.split(key, 19)

    def u(k, lo, hi):
        return 0.5 * (lo + hi) + scale * (jax.random.uniform(k) - 0.5) * (hi - lo)

    wind_dir = jax.random.uniform(ks[0], maxval=2 * jnp.pi)
    wind_speed = scale * jax.random.uniform(ks[1], maxval=3.0)
    return DRParams(
        twr=u(ks[2], *sim.dr_twr),
        rate_tau=u(ks[3], 0.02, 0.07),
        thrust_tau=u(ks[4], 0.015, 0.05),
        drag_lin=u(ks[5], *sim.dr_drag_lin),
        drag_quad=u(ks[6], *sim.dr_drag_quad),
        ang_acc=u(ks[7], 80.0, 250.0),
        yaw_auth=u(ks[8], 0.3, 0.6),
        sag_rate=u(ks[9], 0.0, 0.03),
        wind_mean=wind_speed * jnp.array([jnp.cos(wind_dir), jnp.sin(wind_dir), 0.0]),
        gust_sigma=scale * jax.random.uniform(ks[10], maxval=1.0),
        act_delay=jnp.round(u(ks[11], -0.49, 2.49)).astype(jnp.int32),
        cam_delay=jnp.round(u(ks[12], -0.49, 2.49)).astype(jnp.int32),
        cam_tilt=jnp.deg2rad(u(ks[13], *sim.dr_cam_tilt_deg)),
        tile=u(ks[14], 0.7, 1.5),
        gain=u(ks[15], 0.8, 1.2),
        offset=u(ks[16], -0.05, 0.05),
        pix_noise=u(ks[17], 0.01, 0.05),
        drop_prob=u(ks[18], 0.0, 0.05),
        **sample_imu(jax.random.fold_in(key, 1), scale),
    )


def sample_imu(key, scale=1.0):
    """Typical small-drone MEMS IMU after calibration. Vibration from the motors dominates
    the accelerometer noise in flight. A separate key keeps the other parameters' random
    draws (and so all earlier results) unchanged."""
    k = jax.random.split(key, 5)
    s = 1.0 + scale * (jax.random.uniform(k[0]) - 0.5)          # 0.5 .. 1.5
    return dict(
        imu_scale=s,
        gyro_bias=scale * s * 0.05 * jax.random.normal(k[1], (3,)),   # ~3 deg/s
        gyro_noise=s * 0.02,
        gyro_vib=s * 0.15 * (1.0 + scale * (2 * jax.random.uniform(k[2]) - 1)),  # 0 .. 0.3 rad/s
        accel_bias=scale * s * 0.2 * jax.random.normal(k[3], (3,)),
        accel_noise=s * 0.3,
        accel_vib=s * 1.5 * (1.0 + scale * (2 * jax.random.uniform(k[4]) - 1)),  # 0 .. 3 m/s^2
    )


def with_imu_scale(p: "DRParams", v):
    """Rescale every IMU error term to noise level v (for the IMU-noise sweep)."""
    r = v / p.imu_scale
    return p._replace(imu_scale=jnp.full_like(p.imu_scale, v), gyro_bias=p.gyro_bias * r[..., None],
                      gyro_noise=p.gyro_noise * r, gyro_vib=p.gyro_vib * r,
                      accel_bias=p.accel_bias * r[..., None], accel_noise=p.accel_noise * r,
                      accel_vib=p.accel_vib * r)


# ---------------------------------------------------------------- environment
ACT_DIM = 4
ACT_BUF_LEN = 12   # stick history kept in the state (>= any policy's history and delay)

def reset(key, sim: Sim = Sim(), dr_scale=1.0, level=1.0) -> DroneState:
    ks = jax.random.split(key, 11)
    params = sample_params(ks[0], dr_scale, sim)

    # Start near hover at a random spot, heading and tilt.
    pos = jnp.array([*jax.random.uniform(ks[1], (2,), minval=-50, maxval=50),
                     jax.random.uniform(ks[2], minval=1.0, maxval=3.0)])
    yaw = jax.random.uniform(ks[3], minval=-jnp.pi, maxval=jnp.pi)
    axis = jax.random.normal(ks[4], (3,)) * jnp.array([1.0, 1.0, 0.0])
    axis = axis / (jnp.linalg.norm(axis) + 1e-8)
    tilt = jax.random.uniform(ks[5], maxval=jnp.deg2rad(25.0))
    q_tilt = jnp.concatenate([jnp.cos(tilt / 2)[None], jnp.sin(tilt / 2) * axis])
    q_yaw = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
    quat = quat_mul(q_yaw, q_tilt)

    # Target somewhere ahead, inside the camera's field of view. With a curriculum, the
    # ranges move from the easy ones (level 0) to the full ones (level 1); distances
    # interpolate geometrically so the early levels aren't skipped over.
    (d_lo, d_hi), (a_lo, a_hi), b_max = sim.target_dist, sim.target_alt, sim.target_bearing_deg
    if sim.curriculum and sim.level_mix:
        # Mixed difficulty (skewed toward hard), so easy flights stay in the data.
        level = level * jnp.sqrt(jax.random.uniform(ks[9]))
    if sim.curriculum:
        d_lo = sim.easy_dist[0] * (d_lo / sim.easy_dist[0]) ** level
        d_hi = sim.easy_dist[1] * (d_hi / sim.easy_dist[1]) ** level
        a_lo = sim.easy_alt[0] + (a_lo - sim.easy_alt[0]) * level
        a_hi = sim.easy_alt[1] + (a_hi - sim.easy_alt[1]) * level
        b_max = sim.easy_bearing_deg + (b_max - sim.easy_bearing_deg) * level
    bearing = yaw + jnp.deg2rad(b_max * jax.random.uniform(ks[6], minval=-1.0, maxval=1.0))
    dist = jax.random.uniform(ks[7], minval=d_lo, maxval=d_hi)
    target = jnp.array([pos[0] + dist * jnp.cos(bearing), pos[1] + dist * jnp.sin(bearing),
                        jax.random.uniform(ks[8], minval=a_lo, maxval=a_hi)])

    hover_stick = 2.0 / params.twr - 1.0
    act_buf = jnp.tile(jnp.array([hover_stick, 0.0, 0.0, 0.0]), (max(sim.max_act_delay + 1, ACT_BUF_LEN), 1))
    s = DroneState(
        pos=pos,
        vel=jax.random.normal(ks[9], (3,)) * 0.5,
        quat=quat,
        omega=jax.random.normal(ks[10], (3,)) * 0.5,
        thrust=jnp.array(sim.gravity),
        act_buf=act_buf,
        gust=jnp.zeros(3),
        target=target,
        prev_dist=jnp.linalg.norm(target - pos),
        d0=jnp.linalg.norm(target - pos),
        level=jnp.asarray(level, jnp.float32),
        gaze=jnp.zeros(2),
        imu_buf=jnp.tile(jnp.concatenate([jnp.zeros(3), quat_to_rotmat(quat)[2, :] * sim.gravity / 20.0]),
                         (sim.imu_hist, 1)),
        # Before the first sighting, all a pilot knows is the prior: ahead, at the
        # middle of the distance and altitude ranges.
        belief=jnp.array([pos[0] + 0.5 * (d_lo + d_hi) * jnp.cos(yaw), pos[1] + 0.5 * (d_lo + d_hi) * jnp.sin(yaw),
                          0.5 * (a_lo + a_hi)]),
        seen=jnp.array(0.0),
        since_seen=jnp.array(0.0),
        t=jnp.array(0),
        params=params,
    )
    return update_belief(s, sim, 0.0) if sim.vis_teacher else s


def target_visible(s: DroneState, sim: Sim):
    """Is the target inside the wide camera view and within `vis_range`?"""
    c, sn = jnp.cos(s.params.cam_tilt), jnp.sin(s.params.cam_tilt)
    uptilt = jnp.array([[c, 0.0, -sn], [0.0, 1.0, 0.0], [sn, 0.0, c]])
    w = (quat_to_rotmat(s.quat) @ uptilt).T @ (s.target - s.pos)
    tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[0]) / 2)
    tan_v = tan_h * sim.img_h / sim.img_w
    x = jnp.maximum(w[0], 1e-3)
    in_view = (w[0] > 0) & (jnp.abs(w[1] / x) <= tan_h) & (jnp.abs(w[2] / x) <= tan_v)
    return in_view & (jnp.linalg.norm(s.target - s.pos) <= sim.vis_range)


def update_belief(s: DroneState, sim: Sim, dt) -> DroneState:
    vis = target_visible(s, sim)
    return s._replace(belief=jnp.where(vis, s.target, s.belief), seen=jnp.maximum(s.seen, vis.astype(jnp.float32)),
                      since_seen=jnp.where(vis, 0.0, s.since_seen + dt))


def physics_substep(key, s: DroneState, applied, wind, sim: Sim) -> DroneState:
    p = s.params
    k_thrust, k_vib = jax.random.split(key)
    t_sec = s.t * sim.ctrl_dt

    # Throttle -> thrust. The battery sags over time and the motors lag.
    max_thrust = p.twr * sim.gravity * jnp.maximum(sim.sag_floor, 1.0 - p.sag_rate * t_sec)
    thrust_cmd = (applied[0] + 1.0) / 2.0 * max_thrust
    thrust = s.thrust + (thrust_cmd - s.thrust) * jnp.minimum(sim.dt / p.thrust_tau, 1.0)

    # Ground effect: extra lift close to the ground. Plus prop/turbulence noise.
    ground_boost = 1.0 + 0.3 * jnp.exp(-jnp.maximum(s.pos[2], 0.0) / 0.2)
    thrust_eff = thrust * ground_boost * (1.0 + 0.03 * jax.random.normal(k_thrust))

    # Rates: the FC tracks stick commands, limited by motor torque. Near full throttle
    # the motors saturate, so there is little left for rotating (no "headroom").
    headroom = jnp.clip((1.0 - thrust / max_thrust) * 5.0, 0.2, 1.0)
    omega_cmd = applied[1:] * sim.max_rate
    alpha = (omega_cmd - s.omega) / p.rate_tau
    alpha_lim = p.ang_acc * headroom * jnp.array([1.0, 1.0, p.yaw_auth])
    alpha = jnp.clip(alpha, -alpha_lim, alpha_lim) + 5.0 * jax.random.normal(k_vib, (3,))
    omega = s.omega + alpha * sim.dt

    # Translation: thrust along body z, gravity, and drag relative to the moving air.
    R = quat_to_rotmat(s.quat)
    air_vel = s.vel - wind
    acc = (R @ jnp.array([0.0, 0.0, thrust_eff]) - jnp.array([0.0, 0.0, sim.gravity])
           - p.drag_lin * air_vel - p.drag_quad * jnp.linalg.norm(air_vel) * air_vel)
    vel = s.vel + acc * sim.dt
    pos = s.pos + vel * sim.dt

    q = s.quat + 0.5 * quat_mul(s.quat, jnp.concatenate([jnp.zeros(1), omega])) * sim.dt
    q = q / jnp.linalg.norm(q)
    return s._replace(pos=pos, vel=vel, quat=q, omega=omega, thrust=thrust)


def imu_measure(key, s_prev: DroneState, s: DroneState, sim: Sim) -> jax.Array:
    """One IMU sample at the end of a control step, scaled to ~[-1, 1].
    Gyro: body rates + bias + noise that grows with throttle (vibration).
    Accelerometer: *specific force* R^T (a + g), not gravity. In accelerated flight it
    mostly reads the thrust direction, so it can't simply be read as "which way is down".
    Both saturate like real sensors (2000 deg/s, 16 g)."""
    p = s.params
    kg, ka = jax.random.split(key)
    R = quat_to_rotmat(s.quat)
    accel_world = (s.vel - s_prev.vel) / sim.ctrl_dt
    specific_force = R.T @ (accel_world + jnp.array([0.0, 0.0, sim.gravity]))
    throttle = jnp.clip(s.thrust / (p.twr * sim.gravity), 0.0, 1.0)
    gyro = s.omega + p.gyro_bias + (p.gyro_noise + p.gyro_vib * throttle) * jax.random.normal(kg, (3,))
    accel = specific_force + p.accel_bias + (p.accel_noise + p.accel_vib * throttle) * jax.random.normal(ka, (3,))
    gyro = jnp.clip(gyro, -34.9, 34.9)
    accel = jnp.clip(accel, -16 * sim.gravity, 16 * sim.gravity)
    return jnp.concatenate([gyro / sim.max_rate, accel / 20.0])


def progress_gain(s: DroneState, sim: Sim):
    """Reward per metre of progress. Long range: 20 for the whole approach, so the
    reward scale doesn't grow with distance."""
    return 20.0 / jnp.maximum(s.d0, 1.0) if sim.progress_norm else 2.0


def step(key, s: DroneState, action, sim: Sim = Sim(), dr_scale=1.0, level=1.0):
    """One 50 Hz control step. Returns (next_state, reward, done, info), auto-resetting."""
    action = jnp.clip(action, -1.0, 1.0)
    p = s.params
    k_gust, k_phys, k_reset = jax.random.split(key, 3)

    # Latency: the drone executes the command from `act_delay` steps ago.
    act_buf = jnp.concatenate([action[None], s.act_buf[:-1]])
    applied = act_buf[p.act_delay]

    # Wind = steady part + gusts (an OU process with a 1 s correlation time).
    dt = sim.ctrl_dt
    gust = s.gust - s.gust * dt + p.gust_sigma * jnp.sqrt(2 * dt) * jax.random.normal(k_gust, (3,))
    wind = p.wind_mean + gust

    s_next = s
    for k in jax.random.split(k_phys, sim.substeps):
        s_next = physics_substep(k, s_next, applied, wind, sim)

    # ---- reward: get there fast, keep the target in view, fly smoothly
    to_target = s_next.target - s_next.pos
    dist = jnp.linalg.norm(to_target)
    R = quat_to_rotmat(s_next.quat)
    facing = jnp.dot(R @ camera_axis_body(p.cam_tilt), to_target / dist)
    reward = (
        progress_gain(s, sim) * (s.prev_dist - dist)     # progress toward target
        - sim.time_penalty                               # time penalty: be quick
        + sim.w_perception * facing                      # keep target on camera
        - 0.0005 * jnp.sum(s_next.omega ** 2)
        - 0.02 * jnp.sum((action - s.act_buf[0]) ** 2)
    )
    success = dist < sim.goal_radius
    crashed = (s_next.pos[2] < 0.0) | (s_next.pos[2] > sim.ceiling) | (dist > jnp.maximum(sim.max_dist, 1.5 * s.d0))
    t = s.t + 1
    timeout = t >= sim.max_steps
    reward = reward + 10.0 * success - 10.0 * crashed
    done = success | crashed | timeout

    imu = imu_measure(jax.random.fold_in(key, 7), s, s_next, sim)
    s_next = s_next._replace(act_buf=act_buf, gust=gust, prev_dist=dist, t=t,
                             imu_buf=jnp.concatenate([imu[None], s_next.imu_buf[:-1]]))
    if sim.vis_teacher:
        s_next = update_belief(s_next, sim, dt)
    info = {"success": success, "crashed": crashed & ~success, "timeout": timeout & ~success & ~crashed,
            "time": t * dt, "speed": jnp.linalg.norm(s_next.vel), "dist": dist}

    s_reset = reset(k_reset, sim, dr_scale, level)
    s_out = jax.tree.map(lambda a, b: jnp.where(done, a, b), s_reset, s_next)
    return s_out, reward, done, info


# ---------------------------------------------------------------- observations
def teacher_obs(s: DroneState, sim: Sim = Sim()) -> jax.Array:
    """Privileged, noise-free state in the body frame. Only exists in simulation."""
    R = quat_to_rotmat(s.quat)
    rel_b = R.T @ ((s.belief if sim.vis_teacher else s.target) - s.pos)
    if sim.obs_v2:
        # Long range: raw metres would be far out of scale, so split the target into a
        # direction, a log distance and a near-field offset; add altitude (ground!).
        d = jnp.linalg.norm(rel_b)
        return jnp.concatenate([
            rel_b / (d + 1e-6), jnp.log1p(d)[None] / 6.0, rel_b * jnp.minimum(1.0, 5.0 / (d + 1e-6)) / 5.0,
            (R.T @ s.vel) / 20.0, R[2, :], s.omega / sim.max_rate, s.act_buf[:sim.hist_len].reshape(-1),
            s.params.cam_tilt[None], s.pos[2:3] / 50.0,
        ] + ([jnp.stack([s.seen, s.seen * (s.since_seen == 0), jnp.minimum(s.since_seen, 10.0) / 10.0])]
             if sim.vis_teacher else []))
    return jnp.concatenate([
        rel_b / 5.0, (R.T @ s.vel) / 5.0, R[2, :],          # R[2,:] = gravity direction in body
        s.omega / sim.max_rate, s.act_buf[:sim.hist_len].reshape(-1), s.params.cam_tilt[None],
    ])


def critic_obs(s: DroneState, sim: Sim = Sim()) -> jax.Array:
    """Teacher obs plus the hidden randomisation and time left. Used only by the value
    function, which never has to run on the real drone ("asymmetric actor-critic")."""
    p = s.params
    R = quat_to_rotmat(s.quat)
    wind_b = R.T @ (p.wind_mean + s.gust)
    extra = jnp.stack([
        p.twr / 5.0, p.rate_tau * 20, p.thrust_tau * 30, p.drag_lin, p.drag_quad * 10,
        p.ang_acc / 200.0, p.yaw_auth, p.sag_rate * 30, p.gust_sigma,
        p.act_delay / 2.0, 1.0 - s.t / sim.max_steps, s.level,
    ])
    obs = [teacher_obs(s, sim), wind_b / 3.0, extra]
    if sim.vis_teacher:   # the critic still knows where the target really is
        rel = R.T @ (s.target - s.pos)
        d = jnp.linalg.norm(rel)
        obs.append(jnp.concatenate([rel / (d + 1e-6), jnp.log1p(d)[None] / 6.0]))
    return jnp.concatenate(obs)


def long_range_sim() -> Sim:
    """Long-range task: targets 100-200 m away at 0-100 m altitude, 1 m success radius,
    a faster drone (~40 m/s top speed), foveated camera, physically correct target size."""
    return Sim(
        max_steps=750, goal_radius=1.0, target_radius=1.0,
        target_dist=(100.0, 200.0), target_alt=(0.0, 100.0), target_bearing_deg=15.0,
        ceiling=150.0, max_dist=25.0,
        curriculum=True, easy_dist=(4.0, 12.0), easy_alt=(1.0, 4.0), easy_bearing_deg=40.0,
        progress_norm=True, obs_v2=True, time_penalty=0.02, w_perception=0.01,
        dr_twr=(4.0, 7.0), dr_drag_lin=(0.1, 0.3), dr_drag_quad=(0.015, 0.04),
        dr_cam_tilt_deg=(30.0, 60.0),
        views_hfov=(120.0, 30.0, 8.0), area_target=True, fog_dist=600.0, terrain_layers=True,
    )
