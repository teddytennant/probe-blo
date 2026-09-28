"""Algorithm 2: perturbed gradient descent on the lower-level objective.

PGD(K, x, y^0, β, r, ϱ, ΔG, ΔK):
    τ ← -ΔK - 1
    for k = 0..K-1:
        if ||∇_y g(x, y^k)|| <= ϱ and k - τ > ΔK:
            ỹ ← y^k; τ ← k
            ξ ~ Unif(B_0(r)); y^k ← y^k + ξ
        if k - τ == ΔK and g(x, y^k) - g(x, ỹ) > -ΔG:
            return ỹ
        y^{k+1} ← y^k - β ∇_y g(x, y^k)
    return y^K
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probe_blo.toy import fig1_oracles, grad_y_g as toy_grad_y_g, g as toy_g


def uniform_in_ball(key, radius, like):
    """ξ ~ Unif(B_0(r)) in the Euclidean ball of the same shape as `like`.

    Direction ~ N(0, I) then normalize; radius = r * U^{1/q} with U ~ Unif[0, 1]
    and q = dim(y). For q = 1 this is Unif[-r, r].
    """
    y = jnp.asarray(like)
    r = jnp.asarray(radius, dtype=y.dtype)
    q = int(y.size)
    key_dir, key_u = jax.random.split(key)
    z = jax.random.normal(key_dir, y.shape, dtype=y.dtype)
    z_flat = jnp.reshape(z, (q,))
    nrm = jnp.linalg.norm(z_flat)
    unit_flat = jnp.where(nrm > 0, z_flat / nrm, jnp.zeros_like(z_flat))
    u = jax.random.uniform(key_u, (), dtype=y.dtype)
    rad = r * (u ** (jnp.array(1.0, dtype=y.dtype) / q))
    return jnp.reshape(unit_flat * rad, y.shape)


def pgd(
    K,
    x,
    y0,
    beta,
    r,
    varrho,
    delta_G,
    delta_K,
    key,
    g=None,
    grad_y_g=None,
    oracles=None,
):
    """Algorithm 2 exactly, with jax.random key-splitting for the ball noise.

    If the loop finishes without the SOSP early return, returns y^K.
    """
    if oracles is None:
        oracles = fig1_oracles()
    if g is None:
        g = oracles.g
    if grad_y_g is None:
        grad_y_g = oracles.grad_y_g

    K = int(K)
    delta_K = int(delta_K)
    beta = float(beta)
    r = float(r)
    varrho = float(varrho)
    delta_G = float(delta_G)

    x = jnp.asarray(x)
    y = jnp.asarray(y0)
    tau = -delta_K - 1
    y_tilde = y
    for k in range(K):
        key, sub = jax.random.split(key)
        gy = jnp.asarray(grad_y_g(x, y))
        gnorm = float(jnp.linalg.norm(jnp.reshape(gy, (-1,))))
        if gnorm <= varrho and (k - tau > delta_K):
            y_tilde = y
            tau = k
            xi = uniform_in_ball(sub, r, y)
            y = y + xi
        if (k - tau == delta_K) and (float(g(x, y) - g(x, y_tilde)) > -delta_G):
            return y_tilde
        y = y - beta * jnp.asarray(grad_y_g(x, y))
    return y


# Default Figure 1 bindings, used if a caller wants the closed forms by name.
g = toy_g
grad_y_g = toy_grad_y_g
