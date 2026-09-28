"""Algorithm 1: PROBE (Perturbed gRadient descent On Bilevel problems with Escape).

Input T, K_t, N_t, x0, y0, v0, α_t, β_t, μ_t, and PGD params r_t, ϱ_t, ΔG,t, ΔK,t
for t = 0..T-1:
    y_{t+1} ← PGD(K_t, x_t, y_t, β_t, r_t, ϱ_t, ΔG,t, ΔK,t)
    Solve (∇²_{yy} g(x_t, y_{t+1}) + μ_t I) v = ∇_y f(x_t, y_{t+1})
        from v_t by N_t-step CG with warm start → v_{t+1}
    x_{t+1} ← x_t − α_t ∇Φ̂_μ(x_t)
    where ∇Φ̂_μ(x_t) = ∇_x f(x_t, y_{t+1}) − ∇²_{xy} g(x_t, y_{t+1}) v_{t+1}
return (x_T, y_T)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probe_blo.cg import cg
from probe_blo.pgd import pgd
from probe_blo.toy import add_mu_I, apply_linear, fig1_oracles


def as_schedule(value):
    """Treat a scalar as a constant schedule, or pass callables of t through."""
    if callable(value):
        return value
    return lambda t, _v=value: _v


def default_mu_schedule(mu0=1.0, p=0.5):
    """μ_t = μ0 / (t+1)^p."""
    mu0 = float(mu0)
    p = float(p)
    return lambda t: mu0 / ((t + 1) ** p)


def hypergradient(x, y, v, oracles):
    """μ-augmented hypergradient ∇Φ̂_μ(x) = ∇_x f(x,y) − ∇²_{xy} g(x,y) v."""
    gx = jnp.asarray(oracles.grad_x_f(x, y))
    hxy = jnp.asarray(oracles.hess_xy_g(x, y))
    return gx - apply_linear(hxy, v)


def probe(
    x0,
    y0,
    T,
    K=50,
    N=5,
    v0=None,
    alpha=0.05,
    beta=0.1,
    mu=None,
    r=0.1,
    varrho=1e-3,
    delta_G=1e-4,
    delta_K=10,
    key=None,
    oracles=None,
    cg_tol=1e-10,
):
    """Algorithm 1. Schedules may be scalars or callables of t.

    Defaults: constant α, β, K, N, r, ϱ, ΔG, ΔK and μ_t = 1 / (t+1)^{1/2}.
    Returns trajectories of x, y, v, and the hypergradient.
    """
    if oracles is None:
        oracles = fig1_oracles()
    if key is None:
        key = jax.random.PRNGKey(0)
    if mu is None:
        mu = default_mu_schedule()

    sched_K = as_schedule(K)
    sched_N = as_schedule(N)
    sched_alpha = as_schedule(alpha)
    sched_beta = as_schedule(beta)
    sched_mu = as_schedule(mu)
    sched_r = as_schedule(r)
    sched_varrho = as_schedule(varrho)
    sched_dG = as_schedule(delta_G)
    sched_dK = as_schedule(delta_K)

    x = jnp.asarray(x0)
    y = jnp.asarray(y0)
    if v0 is None:
        v = jnp.zeros_like(y)
    else:
        v = jnp.asarray(v0)

    xs = [x]
    ys = [y]
    vs = [v]
    hgs = []

    T = int(T)
    for t in range(T):
        Kt = int(sched_K(t))
        Nt = int(sched_N(t))
        alpha_t = float(sched_alpha(t))
        beta_t = float(sched_beta(t))
        mu_t = float(sched_mu(t))
        r_t = float(sched_r(t))
        varrho_t = float(sched_varrho(t))
        dG_t = float(sched_dG(t))
        dK_t = int(sched_dK(t))

        key, k_pgd = jax.random.split(key)
        y = pgd(
            Kt,
            x,
            y,
            beta_t,
            r_t,
            varrho_t,
            dG_t,
            dK_t,
            k_pgd,
            oracles=oracles,
        )

        b = jnp.asarray(oracles.grad_y_f(x, y))
        Hmu = add_mu_I(oracles.hess_yy_g(x, y), mu_t)

        def hvp(z, _H=Hmu):
            return apply_linear(_H, z)

        v = cg(hvp, b, v, Nt, tol=cg_tol)
        hg = hypergradient(x, y, v, oracles)
        x = x - alpha_t * hg

        xs.append(x)
        ys.append(y)
        vs.append(v)
        hgs.append(hg)

    hgs_arr = jnp.stack(hgs) if hgs else jnp.array([])
    return {
        "x": x,
        "y": y,
        "v": v,
        "xs": jnp.stack(xs),
        "ys": jnp.stack(ys),
        "vs": jnp.stack(vs),
        "hypergrads": hgs_arr,
        "hypergrad": hgs[-1] if hgs else jnp.zeros_like(x),
    }
