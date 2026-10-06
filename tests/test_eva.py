"""Seeded CPU checks for the empirical VAE ELBO, causality, and toy fit."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from eva.data import make_batch, mode_centers, toy_config
from eva.model import (
    OBS_STD,
    ancestral_sample,
    decode,
    elbo_parts,
    init_params,
    mean_kl,
    train,
)

N_STEPS = 400
BATCH = 128
HIDDEN = 32


@pytest.fixture(scope="module")
def trained_models():
    cfg = toy_config()
    key = jax.random.PRNGKey(0)
    eva = init_params(key, cfg, hidden=HIDDEN, empirical=True)
    gauss = init_params(key, cfg, hidden=HIDDEN, empirical=False)
    eva = train(eva, True, N_STEPS, BATCH, seed=1, cfg=cfg)
    gauss = train(gauss, False, N_STEPS, BATCH, seed=1, cfg=cfg)
    return {"eva": eva, "gauss": gauss, "cfg": cfg}


def test_elbo_matches_hand_expanded_gaussian_kl():
    cfg = toy_config()
    key = jax.random.PRNGKey(11)
    k_init, k_data, k_elbo = jax.random.split(key, 3)
    params = init_params(k_init, cfg, hidden=HIDDEN, empirical=True)
    # Non-zero linear prior so a standard-normal prior would not match by accident.
    params["prior_w"] = params["prior_w"].at[:, :].set(0.35)
    params["prior_b"] = params["prior_b"] + 0.4
    params["prior_b0"] = params["prior_b0"] + jnp.array([-0.2])
    params["prior_log_std"] = params["prior_log_std"] + jnp.array([0.15, -0.25])

    x = make_batch(k_data, 16, cfg)
    value, parts = elbo_parts(params, x, k_elbo, empirical=True)

    mu = parts["mu"]
    q_var = parts["std"] ** 2
    p_mean = parts["prior_mean"]
    p_var = parts["prior_var"]
    x_mean = parts["x_mean"]
    obs_var = parts["obs_var"]

    # Closed form KL(q || p) for diagonal Gaussians. Subtracted in the ELBO.
    kl = 0.5 * (
        jnp.log(p_var)
        - jnp.log(q_var)
        - 1.0
        + q_var / p_var
        + (mu - p_mean) ** 2 / p_var
    )
    # Gaussian log density of x under Normal(dec(z_1:t), sigma_x^2).
    log_px = -0.5 * (
        jnp.log(2.0 * jnp.pi * obs_var) + (x - x_mean) ** 2 / obs_var
    )
    hand = (log_px.sum(axis=(-1, -2)) - kl.sum(axis=(-1, -2))).mean()
    assert jnp.allclose(value, hand, atol=1e-8, rtol=1e-8)

    z = parts["z"]
    mean0 = jnp.broadcast_to(params["prior_b0"], z[:, 0, :].shape)
    mean1 = z[:, 0, :] @ params["prior_w"] + params["prior_b"]
    assert jnp.allclose(p_mean[:, 0, :], mean0, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(p_mean[:, 1, :], mean1, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(obs_var, OBS_STD ** 2, atol=1e-12, rtol=0.0)
    # The conditional prior used above is not N(0, I).
    assert float(jnp.max(jnp.abs(p_mean))) > 1e-3
    assert float(jnp.max(jnp.abs(p_var - 1.0))) > 1e-3


def test_decoder_mean_ignores_future_latents():
    cfg = toy_config()
    # T = 3 so more than one future index is checked. Toy training uses T = 2.
    cfg = dict(cfg)
    cfg["T"] = 3
    params = init_params(jax.random.PRNGKey(3), cfg, hidden=HIDDEN, empirical=True)
    z = jax.random.normal(jax.random.PRNGKey(4), (8, 3, cfg["latent_dim"]), dtype=jnp.float64)
    base = decode(params, z)
    for t_future in range(1, 3):
        altered = z.at[:, t_future, :].add(4.5)
        out = decode(params, altered)
        for t in range(t_future):
            assert jnp.allclose(base[:, t, :], out[:, t, :], atol=1e-12, rtol=0.0)
        assert not jnp.allclose(base[:, t_future, :], out[:, t_future, :], atol=1e-6, rtol=0.0)


def test_eva_kl_smaller_than_standard_normal_prior(trained_models):
    cfg = trained_models["cfg"]
    x = make_batch(jax.random.PRNGKey(21), 512, cfg)
    kl_eva = mean_kl(trained_models["eva"], x, jax.random.PRNGKey(22), empirical=True)
    kl_gauss = mean_kl(
        trained_models["gauss"], x, jax.random.PRNGKey(22), empirical=False
    )
    kl_eva = float(kl_eva)
    kl_gauss = float(kl_gauss)
    assert kl_gauss > 1.5, f"gaussian prior KL collapsed: {kl_gauss}"
    assert kl_eva < kl_gauss - 1.0, f"eva KL {kl_eva} vs gaussian KL {kl_gauss}"


def test_ancestral_samples_nearer_training_modes(trained_models):
    cfg = trained_models["cfg"]
    centers = mode_centers(cfg)
    n = 4000
    x_eva, _ = ancestral_sample(
        trained_models["eva"],
        jax.random.PRNGKey(30),
        n,
        cfg["T"],
        cfg["latent_dim"],
        empirical=True,
        sample_obs=True,
    )
    x_gauss, _ = ancestral_sample(
        trained_models["gauss"],
        jax.random.PRNGKey(31),
        n,
        cfg["T"],
        cfg["latent_dim"],
        empirical=False,
        sample_obs=True,
    )

    def nearest_stats(samples):
        # samples (n, T, obs), centers (modes, T, obs)
        delta = samples[:, None, :, :] - centers[None, :, :, :]
        dist = jnp.sqrt(jnp.sum(delta ** 2, axis=(-1, -2)))
        nearest = dist.min(axis=1)
        # Radius sits above within-mode noise and below half the mode gap.
        radius = 1.5
        return float(nearest.mean()), float(jnp.mean(nearest < radius))

    dist_eva, frac_eva = nearest_stats(x_eva)
    dist_gauss, frac_gauss = nearest_stats(x_gauss)
    assert dist_eva < dist_gauss - 0.4, (
        f"distance eva {dist_eva} gauss {dist_gauss}; "
        f"fraction eva {frac_eva} gauss {frac_gauss}"
    )
    assert frac_eva > frac_gauss + 0.05, (
        f"fraction within mode radius eva {frac_eva} gauss {frac_gauss}; "
        f"distance eva {dist_eva} gauss {dist_gauss}"
    )
