"""Toy multimodal sequence from the empirical VAE setup.

z0 ~ Normal(0, sigma0^2)
z1 = a * z0 + b + c_m + eps
x_t = P z_t + offset + eps

Modes are offsets of several transition-noise standard deviations. T = 2.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def toy_config():
    """Return the fixed toy process as a dict of Python scalars and arrays."""
    latent_dim = 1
    obs_dim = 2
    # Two modes, symmetric, far from the origin relative to process noise.
    offsets = jnp.array([[-4.0], [4.0]], dtype=jnp.float64)
    # x = P z + offset. First row is an identity copy of z so the mode is visible.
    observation_map = jnp.array([[1.0], [0.5]], dtype=jnp.float64)
    offset = jnp.array([0.25, -0.15], dtype=jnp.float64)
    return {
        "T": 2,
        "latent_dim": latent_dim,
        "obs_dim": obs_dim,
        "offsets": offsets,
        "observation_map": observation_map,
        "offset": offset,
        "a": 0.5,
        "b": 0.2,
        "sigma0": 0.4,
        "sigma_z": 0.2,
        "sigma_x_data": 0.1,
    }


def make_batch(rng, n, cfg=None):
    """Sample a batch of sequences, shape (n, T, obs_dim)."""
    if cfg is None:
        cfg = toy_config()
    k_mode, k_z0, k_trans, k_obs = jax.random.split(rng, 4)
    n_modes = cfg["offsets"].shape[0]
    mode = jax.random.randint(k_mode, (n,), 0, n_modes)
    c = cfg["offsets"][mode]
    latent_dim = cfg["latent_dim"]
    z0 = cfg["sigma0"] * jax.random.normal(k_z0, (n, latent_dim), dtype=jnp.float64)
    trans = cfg["sigma_z"] * jax.random.normal(k_trans, (n, latent_dim), dtype=jnp.float64)
    z1 = cfg["a"] * z0 + cfg["b"] + c + trans
    z = jnp.stack([z0, z1], axis=1)
    obs_dim = cfg["obs_dim"]
    noise = cfg["sigma_x_data"] * jax.random.normal(
        k_obs, (n, cfg["T"], obs_dim), dtype=jnp.float64
    )
    x = z @ cfg["observation_map"].T + cfg["offset"] + noise
    return x


def mode_centers(cfg=None):
    """Expected observation sequence for each mode, shape (n_modes, T, obs_dim).

    z0 is mean-zero for every mode. The mode offset enters only at the second step.
    """
    if cfg is None:
        cfg = toy_config()
    offsets = cfg["offsets"]
    n_modes, latent_dim = offsets.shape
    z0 = jnp.zeros((n_modes, latent_dim), dtype=jnp.float64)
    z1 = cfg["a"] * z0 + cfg["b"] + offsets
    z = jnp.stack([z0, z1], axis=1)
    return z @ cfg["observation_map"].T + cfg["offset"]
