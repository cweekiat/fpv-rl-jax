"""The pilot's view: a tiny ray-cast FPV camera, written in JAX.

The scene is deliberately simple but has the cues a pilot uses:
  * sky / ground horizon -> attitude
  * checkered ground     -> height and speed (optical flow between frames)
  * dark target ball     -> bearing (position in image) and distance (apparent size)

The image is then corrupted the way a real FPV feed is: exposure and black level
differ per episode, every pixel gets sensor noise, frames arrive late (latency),
and sometimes the link freezes on an old frame.

Foveation (long range): `sim.views_hfov` lists several fields of view of the same
camera, all centred on the optical axis and all rendered at img_w x img_h. A wide
view gives the horizon and motion; narrow ones are like digitally zoomed crops of a
high-resolution sensor, so a distant target still covers some pixels.
"""
import jax
import jax.numpy as jnp

from fpv.env import DroneState, Sim, quat_to_rotmat


def camera_rays(sim: Sim, hfov_deg=None, h=None, w=None):
    """Unit ray direction per pixel in camera coords (x forward, y left, z up)."""
    h, w = h or sim.img_h, w or sim.img_w
    tan_h = jnp.tan(jnp.deg2rad(sim.hfov_deg if hfov_deg is None else hfov_deg) / 2)
    tan_v = tan_h * h / w
    u = (jnp.arange(w) + 0.5) / w * 2 - 1   # -1 left .. 1 right
    v = (jnp.arange(h) + 0.5) / h * 2 - 1   # -1 top  .. 1 bottom
    uu, vv = jnp.meshgrid(u, v)
    d = jnp.stack([jnp.ones_like(uu), -uu * tan_h, -vv * tan_v], axis=-1)
    return d / jnp.linalg.norm(d, axis=-1, keepdims=True)


def _checker(x, y, scale):
    return (jnp.floor(x / scale) + jnp.floor(y / scale)) % 2


def _rot_y_up(a):
    """Rotation that pitches the x axis up by angle a (about y)."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def gaze_direction(gaze, sim: Sim):
    """Unit direction in camera coords of image point `gaze` = (u, v) in the wide view."""
    tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[0]) / 2)
    tan_v = tan_h * sim.img_h / sim.img_w
    d = jnp.array([1.0, -gaze[0] * tan_h, -gaze[1] * tan_v])
    return d / jnp.linalg.norm(d)


def target_gaze(s: DroneState, sim: Sim):
    """Where the target is in the wide view, clipped to the image: the gaze label."""
    p = s.params
    to_cam = (quat_to_rotmat(s.quat) @ _rot_y_up(p.cam_tilt)).T
    # A visibility-limited teacher can only "look" where it believes the target is.
    w = to_cam @ ((s.belief if sim.vis_teacher else s.target) - s.pos)
    tan_h = jnp.tan(jnp.deg2rad(sim.views_hfov[0]) / 2)
    tan_v = tan_h * sim.img_h / sim.img_w
    x = jnp.maximum(w[0], 1e-3)
    return jnp.clip(jnp.array([-w[1] / x / tan_h, -w[2] / x / tan_v]), -1.0, 1.0)


def render_view(s: DroneState, sim: Sim, hfov_deg, gaze=None, h=None, w=None) -> jax.Array:
    """Noise-free (H, W) grayscale image for one field of view, optionally pointed at
    image point `gaze` of the wide view."""
    p = s.params
    rays = camera_rays(sim, hfov_deg, h, w)
    if gaze is not None:
        g = gaze_direction(gaze, sim)
        yaw, pitch = jnp.arctan2(g[1], g[0]), jnp.arcsin(jnp.clip(g[2], -1.0, 1.0))
        cy, sy = jnp.cos(yaw), jnp.sin(yaw)
        rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        rays = rays @ (rz @ _rot_y_up(pitch)).T
    d = rays @ (quat_to_rotmat(s.quat) @ _rot_y_up(p.cam_tilt)).T   # (H, W, 3) world rays
    o = s.pos
    pix = jnp.deg2rad(hfov_deg) / (w or sim.img_w)                        # angle per pixel

    # Sky gets brighter toward the zenith.
    sky = 0.65 + 0.2 * jnp.clip(d[..., 2], 0.0, 1.0)

    # Ground plane z=0 with a checker texture, fading into haze with distance.
    hit_ground = d[..., 2] < -1e-3
    t = jnp.where(hit_ground, -o[2] / jnp.minimum(d[..., 2], -1e-3), 1e3)
    hx, hy = o[0] + t * d[..., 0], o[1] + t * d[..., 1]
    fog = jnp.exp(-t / sim.fog_dist)
    if sim.terrain_layers:
        # Two texture scales (fields + fine grass), each faded out once a pixel's
        # ground footprint is larger than its tiles ("mip-mapping"), so high-altitude
        # views show the coarse layer instead of aliasing noise.
        footprint = t * pix / jnp.maximum(-d[..., 2], 0.05)
        fine = 0.15 * jnp.clip(1 - footprint / p.tile, 0, 1) * (_checker(hx, hy, p.tile) - 0.5)
        coarse = 0.12 * jnp.clip(1 - footprint / (25 * p.tile), 0, 1) * (_checker(hx, hy, 25 * p.tile) - 0.5)
        ground = (0.42 + fine + coarse) * fog + 0.6 * (1 - fog)
    else:
        ground = (0.35 + 0.15 * _checker(hx, hy, p.tile)) * fog + 0.6 * (1 - fog)
    img = jnp.where(hit_ground, ground, sky)

    # Target: a dark ball.
    w = s.target - o
    dist = jnp.linalg.norm(w)
    ang = jnp.arccos(jnp.clip(d @ (w / dist), -1.0, 1.0))
    radius = jnp.arcsin(jnp.clip(sim.target_radius / dist, 0.0, 1.0))
    if sim.area_target:
        # Physically correct: an anti-aliased disc, and a sub-pixel target darkens a
        # pixel only by the fraction of it that it covers. Far away it disappears.
        edge = jnp.clip((radius - ang) / pix + 0.5, 0.0, 1.0)
        sub = jnp.minimum(1.0, jnp.pi * radius ** 2 / pix ** 2) * jnp.clip(1 - ang / pix, 0.0, 1.0)
        blob = jnp.maximum(edge, sub)
    else:
        # Short-range task: true disc plus a ~1 px soft edge, so it stays visible as a
        # dot even when smaller than a pixel (optimistic beyond ~10 m at 32 px).
        blob = jnp.exp(-0.5 * (jnp.maximum(ang - radius, 0.0) / 0.06) ** 2)
    return img * (1 - blob)


def render(key, s: DroneState, sim: Sim = Sim()) -> jax.Array:
    """Returns (V, H, W) grayscale images in [0, 1], one per field of view."""
    p = s.params
    # View 0 is the wide view; the narrower ones follow the pilot's gaze.
    views = [render_view(s, sim, fov, None if i == 0 else s.gaze) for i, fov in enumerate(sim.views_hfov)]
    if sim.wide_tiles:
        # The wide view again at 2x resolution, cut into 4 tiles the same size as the
        # other views, so they share the CNN.
        h, w = sim.img_h, sim.img_w
        hi = render_view(s, sim, sim.views_hfov[0], None, 2 * h, 2 * w)
        views += [hi[:h, :w], hi[:h, w:], hi[h:, :w], hi[h:, w:]]
    img = jnp.stack(views)
    # Sensor: exposure, black level, noise.
    img = p.gain * img + p.offset + p.pix_noise * jax.random.normal(key, img.shape)
    return jnp.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------- video link
def buffer_len(sim: Sim):
    return sim.frame_stack + sim.max_cam_delay


def init_frames(frame, sim: Sim = Sim()):
    return jnp.broadcast_to(frame, (buffer_len(sim), *frame.shape))


def update_frames(frames, new_frame, done, drop):
    """Push a frame into the history (newest first). A dropped frame repeats the last
    one, and a fresh episode fills the whole history with its first frame."""
    new_frame = jnp.where(drop, frames[0], new_frame)
    shifted = jnp.concatenate([new_frame[None], frames[:-1]])
    return jnp.where(done, jnp.broadcast_to(new_frame, frames.shape), shifted)


def pilot_view(frames, cam_delay, sim: Sim = Sim()):
    """The `frame_stack` most recent frames the pilot has received, after the video
    latency. Returns (V, H, W, frame_stack)."""
    view = jax.lax.dynamic_slice_in_dim(frames, cam_delay, sim.frame_stack, axis=0)
    return jnp.moveaxis(view, 0, -1)
