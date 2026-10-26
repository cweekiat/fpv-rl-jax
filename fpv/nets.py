"""Tiny neural networks written by hand, so nothing is hidden."""
import math

import jax
import jax.numpy as jnp


def init_mlp(key, sizes, last_scale=1.0):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        w = jax.nn.initializers.orthogonal(scale)(k, (n_in, n_out))
        params.append({"w": w, "b": jnp.zeros(n_out)})
    return params


def mlp(params, x, act=jnp.tanh):
    for layer in params[:-1]:
        x = act(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


# Conv layers: (out_channels, kernel, stride)
CONVS = ((16, 5, 2), (32, 3, 2), (32, 3, 2))


def init_cnn(key, img_shape, extra_dim, out_dim, hidden=256, n_views=1):
    """CNN encoder for the camera frames (shared across views), then an MLP that also
    sees `extra` inputs."""
    h, w, c_in = img_shape
    keys = jax.random.split(key, len(CONVS) + 1)
    convs = []
    for k, (c_out, ks, stride) in zip(keys, CONVS):
        wgt = jax.nn.initializers.he_normal()(k, (ks, ks, c_in, c_out))
        convs.append({"w": wgt, "b": jnp.zeros(c_out)})
        c_in, h, w = c_out, math.ceil(h / stride), math.ceil(w / stride)
    head = init_mlp(keys[-1], [n_views * h * w * c_in + extra_dim, hidden, hidden, out_dim], 0.01)
    return {"convs": convs, "head": head}


def conv2d(x, w, stride):
    """'SAME' strided convolution as im2col + one matmul. XLA's CPU conv kernel gets
    very slow inside lax.scan loops; slicing + matmul doesn't."""
    k = w.shape[0]
    b, h, wd, c = x.shape
    oh, ow = math.ceil(h / stride), math.ceil(wd / stride)
    pad_h = max((oh - 1) * stride + k - h, 0)
    pad_w = max((ow - 1) * stride + k - wd, 0)
    x = jnp.pad(x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
    patches = [x[:, i:i + stride * oh:stride, j:j + stride * ow:stride, :]
               for i in range(k) for j in range(k)]
    patches = jnp.concatenate(patches, axis=-1)            # (B, oh, ow, k*k*C)
    return patches @ w.reshape(k * k * c, -1)


def conv_features(convs, img):
    """img: (B, V, H, W, C) or (B, H, W, C) -> (B, V * flat). Every view goes through
    the same conv layers; their features are concatenated."""
    b = img.shape[0]
    x = img.reshape(-1, *img.shape[-3:]) - 0.5
    for layer, (_, _, stride) in zip(convs, CONVS):
        x = jax.nn.relu(conv2d(x, layer["w"], stride) + layer["b"])
    return x.reshape(b, -1)


def cnn(params, img, extra):
    """img: (B, V, H, W, C) or (B, H, W, C), extra: (B, E) -> (B, out)."""
    x = jnp.concatenate([conv_features(params["convs"], img), extra], axis=-1)
    return mlp(params["head"], x, act=jax.nn.relu)


# ---------------------------------------------------------------- recurrent student
def init_crnn(key, img_shape, extra_dim, out_dim, n_views=1, feat=256, hidden=256):
    """CNN encoder -> GRU -> MLP head. The GRU carries memory across control steps."""
    k_cnn, k_enc, k_x, k_h, k_head = jax.random.split(key, 5)
    base = init_cnn(k_cnn, img_shape, extra_dim, out_dim, hidden=feat, n_views=n_views)
    flat_dim = base["head"][0]["w"].shape[0]          # n_views * flat + extra_dim
    return {
        "convs": base["convs"],
        "enc": init_mlp(k_enc, [flat_dim, feat], last_scale=jnp.sqrt(2.0)),
        "gru": {"wx": jax.nn.initializers.orthogonal()(k_x, (feat, 3 * hidden)),
                "wh": jax.nn.initializers.orthogonal()(k_h, (hidden, 3 * hidden)),
                "b": jnp.zeros(3 * hidden)},
        "head": init_mlp(k_head, [hidden, hidden, out_dim], 0.01),
    }


def crnn_encode(params, img, extra):
    """Per-step input features (no memory), batched: (B, ...) -> (B, feat)."""
    x = jnp.concatenate([conv_features(params["convs"], img), extra], axis=-1)
    return jax.nn.relu(mlp(params["enc"], x))


def gru_cell(p, h, x):
    zx, rx, nx = jnp.split(x @ p["wx"] + p["b"], 3, axis=-1)
    zh, rh, nh = jnp.split(h @ p["wh"], 3, axis=-1)
    z, r = jax.nn.sigmoid(zx + zh), jax.nn.sigmoid(rx + rh)
    n = jnp.tanh(nx + r * nh)
    return (1 - z) * n + z * h


def crnn_step(params, h, feat):
    """One control step from precomputed features: (h, feat) -> (h', out)."""
    h = gru_cell(params["gru"], h, feat)
    return h, mlp(params["head"], h, act=jax.nn.relu)
