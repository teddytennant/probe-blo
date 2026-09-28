"""Figure 1 objectives and closed-form derivatives (arXiv:2609.30501).

BLO (eq 1):
    min_{x,y} f(x,y)  s.t.  y in S(x) := argmin_y g(x,y)

SOSP surrogate (eq 2):
    min f(x,y)  s.t.  ∇_y g(x,y) = 0  and  λ_min(∇²_{yy} g(x,y)) >= 0

Figure 1 (x, y in R):
    f(x,y) = (x - 2)^2 + y^2
    g(x,y) = (1/4) y^4 - (1/2) x y^2
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Oracles(NamedTuple):
    """First- and second-order oracles used by PGD, CG, and PROBE."""

    f: Callable
    g: Callable
    grad_x_f: Callable
    grad_y_f: Callable
    grad_y_g: Callable
    hess_yy_g: Callable
    hess_xy_g: Callable


def f(x, y):
    """Upper-level objective f(x,y) = (x - 2)^2 + y^2."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return (x - 2.0) ** 2 + y**2


def g(x, y):
    """Lower-level objective g(x,y) = (1/4) y^4 - (1/2) x y^2."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return 0.25 * y**4 - 0.5 * x * y**2


def grad_x_f(x, y):
    """∇_x f = 2(x - 2)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return 2.0 * (x - 2.0) + 0.0 * y


def grad_y_f(x, y):
    """∇_y f = 2y."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return 2.0 * y + 0.0 * x


def grad_y_g(x, y):
    """∇_y g = y^3 - x y."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return y**3 - x * y


def hess_yy_g(x, y):
    """∇²_{yy} g = 3 y^2 - x."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return 3.0 * y**2 - x


def hess_xy_g(x, y):
    """∇²_{xy} g = -y  (scalar; p = q = 1)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return -y + 0.0 * x


def mu_hypergrad(x, y, mu, v=None):
    """Closed-form μ-augmented hypergradient on Figure 1.

    (3 y^2 - x + μ) v = 2 y
    ∇Φ̂_μ = 2(x - 2) + y v
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    mu = jnp.asarray(mu)
    h_mu = 3.0 * y**2 - x + mu
    if v is None:
        v = (2.0 * y) / h_mu
    else:
        v = jnp.asarray(v)
    hg = 2.0 * (x - 2.0) + y * v
    return hg, v


def apply_linear(H, v):
    """Matrix-vector product that also accepts 0-d (scalar) Hessians."""
    H = jnp.asarray(H)
    v = jnp.asarray(v)
    if H.ndim == 0 or v.ndim == 0:
        return H * v
    return H @ v


def add_mu_I(H, mu):
    """H + μ I, with H a 0-d Hessian or a (q, q) matrix."""
    H = jnp.asarray(H)
    mu = jnp.asarray(mu)
    if H.ndim == 0:
        return H + mu
    q = H.shape[0]
    return H + mu * jnp.eye(q, dtype=H.dtype)


def lambda_min_yy(x, y, hess_yy=hess_yy_g):
    """λ_min(∇²_{yy} g). For the scalar toy this is ∇²_{yy} g itself."""
    H = jnp.asarray(hess_yy(x, y))
    if H.ndim == 0:
        return H
    return jnp.min(jnp.linalg.eigvalsh(H))


def grad_norm_y(x, y, grad_y=grad_y_g):
    """||∇_y g(x,y)||."""
    gy = jnp.reshape(jnp.asarray(grad_y(x, y)), (-1,))
    return jnp.linalg.norm(gy)


def is_fosp(x, y, eps=0.0, grad_y=grad_y_g):
    """Def 1: y is FOSP if ∇_y g = 0; ϵ-FOSP if ||∇_y g|| <= ϵ."""
    return grad_norm_y(x, y, grad_y=grad_y) <= eps


def is_sosp(x, y, eps=0.0, rho=0.0, grad_y=grad_y_g, hess_yy=hess_yy_g):
    """Def 1: SOSP if ∇_y g = 0 and λ_min(∇²_{yy} g) >= 0.

    ϵ-SOSP if ||∇_y g|| <= ϵ and λ_min(∇²_{yy} g) >= -√(ρ ϵ).
    """
    gnorm = grad_norm_y(x, y, grad_y=grad_y)
    lam = lambda_min_yy(x, y, hess_yy=hess_yy)
    thresh = -jnp.sqrt(jnp.asarray(rho) * jnp.asarray(eps)) if eps > 0 else 0.0
    return (gnorm <= eps) and (lam >= thresh)


def autodiff_oracles(f_fn, g_fn) -> Oracles:
    """Generic oracles via jax.grad / jax.hessian / mixed Jacobian."""
    grad_x_f_ad = jax.grad(f_fn, 0)
    grad_y_f_ad = jax.grad(f_fn, 1)
    grad_y_g_ad = jax.grad(g_fn, 1)
    hess_yy_g_ad = jax.hessian(g_fn, 1)
    # (p, q) Jacobian of ∇_x g with respect to y, so (∇²_{xy} g) v is in x-space.
    hess_xy_g_ad = jax.jacfwd(jax.grad(g_fn, 0), 1)
    return Oracles(
        f=f_fn,
        g=g_fn,
        grad_x_f=grad_x_f_ad,
        grad_y_f=grad_y_f_ad,
        grad_y_g=grad_y_g_ad,
        hess_yy_g=hess_yy_g_ad,
        hess_xy_g=hess_xy_g_ad,
    )


def fig1_oracles() -> Oracles:
    """Closed-form Figure 1 oracles (the default)."""
    return Oracles(
        f=f,
        g=g,
        grad_x_f=grad_x_f,
        grad_y_f=grad_y_f,
        grad_y_g=grad_y_g,
        hess_yy_g=hess_yy_g,
        hess_xy_g=hess_xy_g,
    )


def hypergrad_jvp(x, y, v, f_fn=f, g_fn=g):
    """∇_x f - ∇²_{xy} g v via JVP of ∇_x g in direction v (wrt y)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    v = jnp.asarray(v)
    gx = jax.grad(f_fn, 0)(x, y)
    grad_x_g = jax.grad(g_fn, 0)
    _, hxy_v = jax.jvp(lambda yy: grad_x_g(x, yy), (y,), (v,))
    return gx - hxy_v


def hypergrad_vjp(x, y, v, f_fn=f, g_fn=g):
    """∇_x f - ∇²_{xy} g v via VJP of ∇_y g with cotangent v."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    v = jnp.asarray(v)
    gx = jax.grad(f_fn, 0)(x, y)
    grad_y_g_ad = jax.grad(g_fn, 1)
    _, vjp_fn = jax.vjp(lambda xx: grad_y_g_ad(xx, y), x)
    hxy_v = vjp_fn(v)[0]
    return gx - hxy_v
