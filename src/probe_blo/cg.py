"""N-step warm-start conjugate gradient for the adjoint (Algorithm 1).

Solves (∇²_{yy} g + μ I) v = ∇_y f from a warm start v_prev by N CG steps
with Hessian-vector products.
"""

from __future__ import annotations

import jax.numpy as jnp


def residual(hvp, v, b):
    """b - H v."""
    v = jnp.asarray(v)
    b = jnp.asarray(b)
    return b - hvp(v)


def residual_norm(hvp, v, b):
    r = residual(hvp, v, b)
    return jnp.linalg.norm(jnp.reshape(r, (-1,)))


def cg(hvp, b, v_prev, n_steps, tol=1e-10, return_info=False):
    """N-step CG for H v = b, warm-started at v_prev.

    Stops early when the residual norm is already <= tol (0 effective steps
    if the warm start already solves the system).
    """
    b = jnp.asarray(b)
    v = jnp.asarray(v_prev, dtype=b.dtype)
    if v.shape != b.shape:
        v = jnp.reshape(v, b.shape)
    r = b - hvp(v)
    p = r
    rsold = jnp.real(jnp.vdot(jnp.reshape(r, (-1,)), jnp.reshape(r, (-1,))))
    steps = 0
    n_steps = int(n_steps)
    for _ in range(n_steps):
        if float(rsold) <= float(tol) ** 2:
            break
        Hp = hvp(p)
        denom = jnp.real(jnp.vdot(jnp.reshape(p, (-1,)), jnp.reshape(Hp, (-1,))))
        alpha = rsold / denom
        v = v + alpha * p
        r = r - alpha * Hp
        rsnew = jnp.real(jnp.vdot(jnp.reshape(r, (-1,)), jnp.reshape(r, (-1,))))
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        steps += 1
    if return_info:
        rnorm = jnp.sqrt(jnp.maximum(rsold, 0.0))
        return v, {"steps": steps, "residual_norm": rnorm, "residual": r}
    return v
