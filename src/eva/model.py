"""Empirical VAE and a same-architecture causal VAE with a standard normal prior.

Sequence model, one sequence x_1:T continuous:
  q(z_t | x) = Normal(mu_t(x), softplus(raw_t(x))) independent across t
  p(x_t | z_1:t) = Normal(dec(z_1:t), sigma_x^2)
  EVA prior: p(z_t | z_1:t-1) = Normal(Linear(z_1:t-1), sigma_p,t^2)
  Gaussian-prior baseline: p(z_t) = Normal(0, I)

ELBO, single sample, sum over t of
  log p(x_t | z_1:t) - KL(q(z_t | x) || p(z_t | z_1:t-1))
The KL is closed form. The prior mean at t > 1 is the linear layer applied to the
sampled previous latent. Gradients flow into that linear layer. No stop-gradient.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

# Fixed decoder scale. Small enough that reconstruction wants the latent near
# the observation, large enough that a short Adam run stays finite.
OBS_STD = 0.35


def _softplus_inv(y):
    y = jnp.asarray(y, dtype=jnp.float64)
    return jnp.log(jnp.expm1(y))


def init_params(rng, cfg, hidden=32, empirical=True):
    """Initialize encoder, causal decoder, and (optionally) the linear AR prior.

    Encoder skip weights start at the inverse of the toy observation map so both
    models begin able to read the latent off x. The decoder linear path starts
    at that same map. The residual causal MLP output starts at zero.
    """
    k_enc, k_dec, k_prior = jax.random.split(rng, 3)
    t_steps = cfg["T"]
    obs_dim = cfg["obs_dim"]
    latent_dim = cfg["latent_dim"]
    in_dim = t_steps * obs_dim
    out_dim = t_steps * latent_dim * 2

    k_w1, k_w2 = jax.random.split(k_enc, 2)
    enc_w1 = jax.random.normal(k_w1, (in_dim, hidden), dtype=jnp.float64) * jnp.sqrt(
        2.0 / in_dim
    )
    enc_b1 = jnp.zeros((hidden,), dtype=jnp.float64)
    enc_w2 = jnp.zeros((hidden, out_dim), dtype=jnp.float64)
    enc_b2 = jnp.zeros((out_dim,), dtype=jnp.float64)
    enc_skip = jnp.zeros((in_dim, out_dim), dtype=jnp.float64)

    # Layout of the encoder head, per time: [mu (D), raw_std (D)].
    # Invert the first observation coordinate, which copies z.
    scale0 = cfg["observation_map"][0, 0]
    offset0 = cfg["offset"][0]
    for t in range(t_steps):
        src = t * obs_dim  # first obs dim at this time
        mu_col = t * (2 * latent_dim)
        enc_skip = enc_skip.at[src, mu_col].set(1.0 / scale0)
        enc_b2 = enc_b2.at[mu_col].set(-offset0 / scale0)
        # Mildly peaked posterior. softplus(raw) is the standard deviation.
        enc_b2 = enc_b2.at[mu_col + latent_dim].set(_softplus_inv(0.35))

    k_wz, k_wh = jax.random.split(k_dec, 2)
    dec_hidden = hidden
    dec_wz = jax.random.normal(k_wz, (latent_dim, dec_hidden), dtype=jnp.float64) * 0.01
    dec_wh = jax.random.normal(k_wh, (dec_hidden, dec_hidden), dtype=jnp.float64) * 0.01
    dec_bh = jnp.zeros((dec_hidden,), dtype=jnp.float64)
    dec_wy = jnp.zeros((dec_hidden, obs_dim), dtype=jnp.float64)
    dec_by = jnp.array(cfg["offset"], dtype=jnp.float64)
    dec_wlin = jnp.array(cfg["observation_map"].T, dtype=jnp.float64)

    params = {
        "enc_w1": enc_w1,
        "enc_b1": enc_b1,
        "enc_w2": enc_w2,
        "enc_b2": enc_b2,
        "enc_skip": enc_skip,
        "dec_wz": dec_wz,
        "dec_wh": dec_wh,
        "dec_bh": dec_bh,
        "dec_wy": dec_wy,
        "dec_by": dec_by,
        "dec_wlin": dec_wlin,
    }
    if empirical:
        # One shared linear map, trained from the latents. t = 1 uses a learned
        # bias and an empty prefix. Per-step scale: softplus(raw) is sigma_p,t.
        params["prior_w"] = jnp.zeros((latent_dim, latent_dim), dtype=jnp.float64)
        params["prior_b"] = jnp.zeros((latent_dim,), dtype=jnp.float64)
        params["prior_b0"] = jnp.zeros((latent_dim,), dtype=jnp.float64)
        params["prior_log_std"] = jnp.full((t_steps,), _softplus_inv(1.0), dtype=jnp.float64)
    return params


def encode(params, x):
    """Mean-field encoder. Returns mu and std, each (batch, T, latent)."""
    batch, t_steps, obs_dim = x.shape
    flat = x.reshape(batch, t_steps * obs_dim)
    hidden = jnp.tanh(flat @ params["enc_w1"] + params["enc_b1"])
    raw = hidden @ params["enc_w2"] + flat @ params["enc_skip"] + params["enc_b2"]
    latent_dim = raw.shape[-1] // (2 * t_steps)
    raw = raw.reshape(batch, t_steps, 2 * latent_dim)
    mu = raw[..., :latent_dim]
    std = jax.nn.softplus(raw[..., latent_dim:])
    return mu, std


def decode(params, z):
    """Causal decoder mean. x_t depends on z_1:t only, via a recurrent hidden state.

    A linear map of z_t is added to a causal MLP state. Future latents never
    enter the mean at time t.
    """
    hidden_dim = params["dec_wh"].shape[0]

    def step(h, z_t):
        h_new = jnp.tanh(z_t @ params["dec_wz"] + h @ params["dec_wh"] + params["dec_bh"])
        y = z_t @ params["dec_wlin"] + h_new @ params["dec_wy"] + params["dec_by"]
        return h_new, y

    h0 = jnp.zeros((z.shape[0], hidden_dim), dtype=z.dtype)
    z_time = jnp.swapaxes(z, 0, 1)
    _, y_time = jax.lax.scan(step, h0, z_time)
    return jnp.swapaxes(y_time, 0, 1)


def prior_mean(params, z):
    """Autoregressive prior means. t = 1 is a learned bias. Later steps: z_{t-1} W + b."""
    batch = z.shape[0]
    b0 = jnp.broadcast_to(params["prior_b0"], (batch, params["prior_b0"].shape[0]))
    rest = z[:, :-1, :] @ params["prior_w"] + params["prior_b"]
    return jnp.concatenate([b0[:, None, :], rest], axis=1)


def prior_variance(params, like):
    """Broadcast per-timestep prior variance to like.shape."""
    std = jax.nn.softplus(params["prior_log_std"])
    var = std ** 2
    # var: (T,) -> (batch, T, latent)
    return jnp.broadcast_to(var[None, :, None], like.shape)


def elbo_parts(params, x, rng, empirical):
    """Single-sample ELBO and the tensors a hand expansion needs.

    Returns the mean over the batch of the per-sequence ELBO
    (sum over time of log-likelihood minus closed-form KL).
    """
    mu, std = encode(params, x)
    eps = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
    z = mu + std * eps
    x_mean = decode(params, z)
    if empirical:
        p_mean = prior_mean(params, z)
        p_var = prior_variance(params, mu)
    else:
        p_mean = jnp.zeros_like(mu)
        p_var = jnp.ones_like(mu)
    q_var = std ** 2
    obs_var = jnp.asarray(OBS_STD ** 2, dtype=x.dtype)
    # KL(q || p) for diagonal Gaussians, summed over latent dims inside the caller.
    kl = 0.5 * (
        jnp.log(p_var) - jnp.log(q_var) - 1.0 + q_var / p_var + (mu - p_mean) ** 2 / p_var
    )
    log_px = -0.5 * (jnp.log(2.0 * jnp.pi * obs_var) + (x - x_mean) ** 2 / obs_var)
    per_seq = log_px.sum(axis=(-1, -2)) - kl.sum(axis=(-1, -2))
    value = per_seq.mean()
    parts = {
        "mu": mu,
        "std": std,
        "z": z,
        "x_mean": x_mean,
        "prior_mean": p_mean,
        "prior_var": p_var,
        "q_var": q_var,
        "obs_var": obs_var,
        "kl": kl,
        "log_px": log_px,
    }
    return value, parts


def mean_kl(params, x, rng, empirical):
    """Mean over batch and time of KL(q(z_t | x) || p(z_t | z_<t)).

    Each term is the KL of the full diagonal Gaussian at that time (sum over latent dims).
    The EVA prior mean is conditioned on one sample of previous latents.
    """
    _, parts = elbo_parts(params, x, rng, empirical)
    kl_t = parts["kl"].sum(axis=-1)
    return kl_t.mean()


def ancestral_sample(params, rng, n, t_steps, latent_dim, empirical, sample_obs=True):
    """Draw ancestral sequences from the prior, then the causal decoder.

    EVA: z_1 ~ Normal(bias, sigma_p,1^2), z_t ~ Normal(Linear(z_{t-1}), sigma_p,t^2).
    Gaussian-prior model: z_t ~ Normal(0, I) independent of the past.
    """
    keys = jax.random.split(rng, t_steps + 1)
    zs = []
    z_prev = None
    for t in range(t_steps):
        if empirical:
            if t == 0:
                mean = jnp.broadcast_to(params["prior_b0"], (n, latent_dim))
            else:
                mean = z_prev @ params["prior_w"] + params["prior_b"]
            std = jax.nn.softplus(params["prior_log_std"][t])
        else:
            mean = jnp.zeros((n, latent_dim), dtype=jnp.float64)
            std = 1.0
        z_t = mean + std * jax.random.normal(keys[t], (n, latent_dim), dtype=jnp.float64)
        zs.append(z_t)
        z_prev = z_t
    z = jnp.stack(zs, axis=1)
    x_mean = decode(params, z)
    if not sample_obs:
        return x_mean, z
    obs_noise = OBS_STD * jax.random.normal(keys[t_steps], x_mean.shape, dtype=x_mean.dtype)
    return x_mean + obs_noise, z


def _labels(params):
    labels = {}
    for key in params:
        if key.startswith("prior"):
            labels[key] = "prior"
        elif key.startswith("dec"):
            labels[key] = "dec"
        else:
            labels[key] = "enc"
    return labels


def train(params, empirical, n_steps, batch_size, seed, cfg=None, lr_enc=1e-4, lr_prior=3e-2):
    """Adam on the negative ELBO. Prior step size is larger than encoder and decoder.

    A short run is enough for the linear prior to match q, and a smaller encoder
    step keeps the Gaussian-prior baseline from immediately squashing every mode
    into the origin. Both models see this same schedule. The baseline simply has
    no prior parameters.
    """
    from eva.data import make_batch, toy_config

    if cfg is None:
        cfg = toy_config()
    optimizer = optax.multi_transform(
        {
            "enc": optax.adam(lr_enc),
            "dec": optax.adam(lr_enc),
            "prior": optax.adam(lr_prior),
        },
        _labels(params),
    )
    opt_state = optimizer.init(params)

    def loss_fn(p, x, rng):
        value, _ = elbo_parts(p, x, rng, empirical)
        return -value

    @jax.jit
    def step(p, opt_state, x, rng):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, rng)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    rng = jax.random.PRNGKey(seed)
    for _ in range(n_steps):
        rng, k_data, k_elbo = jax.random.split(rng, 3)
        x = make_batch(k_data, batch_size, cfg)
        params, opt_state, _ = step(params, opt_state, x, k_elbo)
    return params
