"""Lock displayed equations of arXiv:2609.30501 on the Figure 1 scalar toy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from probe_blo.cg import cg, residual_norm
from probe_blo.pgd import pgd
from probe_blo.probe import hypergradient, probe
from probe_blo.toy import (
    autodiff_oracles,
    f,
    fig1_oracles,
    g,
    grad_x_f,
    grad_y_f,
    grad_y_g,
    hess_xy_g,
    hess_yy_g,
    hypergrad_jvp,
    hypergrad_vjp,
    is_fosp,
    is_sosp,
    lambda_min_yy,
    mu_hypergrad,
)


POINTS = [(0.0, 2.0), (3.0, 0.0), (2.0, 1.0), (1.0, 0.0)]


def _to_arrays(x, y):
    return jnp.asarray(x), jnp.asarray(y)


def test_fig1_derivatives_match_jax():
    """Closed forms match jax.grad / jax.hessian at the listed (x, y)."""
    for xv, yv in POINTS:
        x, y = _to_arrays(xv, yv)
        np.testing.assert_allclose(
            grad_x_f(x, y), jax.grad(f, 0)(x, y), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            grad_y_f(x, y), jax.grad(f, 1)(x, y), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            grad_y_g(x, y), jax.grad(g, 1)(x, y), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            hess_yy_g(x, y), jax.hessian(g, 1)(x, y), rtol=1e-5, atol=1e-6
        )
        mixed = jax.jacfwd(jax.grad(g, 0), 1)(x, y)
        np.testing.assert_allclose(hess_xy_g(x, y), mixed, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_x_f(x, y), 2.0 * (x - 2.0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad_y_f(x, y), 2.0 * y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad_y_g(x, y), y**3 - x * y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_yy_g(x, y), 3.0 * y**2 - x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hess_xy_g(x, y), -y, rtol=1e-6, atol=1e-6)


def test_saddle_at_1_0_is_fosp_not_sosp():
    """At (x,y)=(1,0): ∇_y g = 0 and ∇²_{yy} g = -1 < 0, FOSP but not SOSP."""
    x, y = _to_arrays(1.0, 0.0)
    np.testing.assert_allclose(float(grad_y_g(x, y)), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(hess_yy_g(x, y)), -1.0, atol=1e-7)
    assert bool(is_fosp(x, y, eps=0.0))
    assert not bool(is_sosp(x, y, eps=0.0))
    assert float(lambda_min_yy(x, y)) < 0.0


def test_local_min_at_1_1_is_sosp():
    """At (x,y)=(1,1): ∇_y g = 0 and ∇²_{yy} g = 2 > 0, SOSP / local min of g."""
    x, y = _to_arrays(1.0, 1.0)
    np.testing.assert_allclose(float(grad_y_g(x, y)), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(hess_yy_g(x, y)), 2.0, atol=1e-7)
    assert bool(is_fosp(x, y, eps=0.0))
    assert bool(is_sosp(x, y, eps=0.0))
    assert float(lambda_min_yy(x, y)) > 0.0


def test_mu_hypergradient_matches_jvp_vjp():
    """2(x-2)+y v with (3y^2-x+μ)v=2y matches jax JVP/VJP of ∇_x f − ∇²_{xy} g v."""
    mu = 0.3
    for xv, yv in POINTS + [(2.0, 1.0), (1.5, 0.5)]:
        x, y = _to_arrays(xv, yv)
        hg_closed, v = mu_hypergrad(x, y, mu)
        np.testing.assert_allclose(
            float((3.0 * y**2 - x + mu) * v), float(2.0 * y), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            float(hg_closed), float(2.0 * (x - 2.0) + y * v), rtol=1e-6, atol=1e-6
        )
        hg_lin = grad_x_f(x, y) - hess_xy_g(x, y) * v
        np.testing.assert_allclose(hg_closed, hg_lin, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hg_closed, hypergrad_jvp(x, y, v), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hg_closed, hypergrad_vjp(x, y, v), rtol=1e-5, atol=1e-6)
        oracles = fig1_oracles()
        np.testing.assert_allclose(
            hg_closed, hypergradient(x, y, v, oracles), rtol=1e-5, atol=1e-6
        )


def test_cg_spd_2x2_and_warm_start():
    """CG on a 2x2 SPD system recovers the exact solve to ~1e-5.

    Warm start from the solution takes 0 effective steps (residual already small).
    """
    H = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    b = jnp.array([1.0, 2.0])
    v_exact = jnp.linalg.solve(H, b)

    def hvp(z):
        return H @ z

    v0 = jnp.zeros_like(b)
    v_cg, info = cg(hvp, b, v0, n_steps=4, tol=1e-10, return_info=True)
    np.testing.assert_allclose(v_cg, v_exact, rtol=1e-5, atol=1e-5)
    assert float(info["residual_norm"]) < 1e-5
    r0 = float(residual_norm(hvp, v0, b))
    r1 = float(residual_norm(hvp, v_cg, b))
    assert r1 < r0

    v_warm, info_w = cg(hvp, b, v_exact, n_steps=4, tol=1e-5, return_info=True)
    assert info_w["steps"] == 0
    np.testing.assert_allclose(v_warm, v_exact, rtol=1e-5, atol=1e-5)
    assert float(info_w["residual_norm"]) <= 1e-5


def test_pgd_saddle_escape_vs_r0():
    """PGD on g(1, ·) from y=0 with r>0 moves |y| away from 0. With r=0 it stays."""
    x = jnp.asarray(1.0)
    y0 = jnp.asarray(0.0)
    key = jax.random.PRNGKey(0)
    y_pert = pgd(
        K=80,
        x=x,
        y0=y0,
        beta=0.1,
        r=0.2,
        varrho=1e-3,
        delta_G=1e-6,
        delta_K=15,
        key=key,
    )
    y_ctrl = pgd(
        K=80,
        x=x,
        y0=y0,
        beta=0.1,
        r=0.0,
        varrho=1e-3,
        delta_G=1e-6,
        delta_K=15,
        key=key,
    )
    assert float(jnp.abs(y_pert)) > 0.1
    np.testing.assert_allclose(float(y_ctrl), 0.0, atol=1e-6)


def test_pgd_sosp_early_return_at_local_min():
    """Already at a local min: returns the anchor or a nearby point whose g did not drop by ΔG."""
    x = jnp.asarray(1.0)
    y0 = jnp.asarray(1.0)
    g0 = float(g(x, y0))
    key = jax.random.PRNGKey(1)
    y_out = pgd(
        K=20,
        x=x,
        y0=y0,
        beta=0.05,
        r=0.05,
        varrho=1e-3,
        delta_G=1e-3,
        delta_K=2,
        key=key,
    )
    # Anchor return is y=1. Nearby points must not have dropped g by ΔG.
    drop = float(g(x, y_out) - g0)
    assert drop > -1e-3
    assert abs(float(y_out) - 1.0) < 0.2


def test_probe_one_outer_step_finite():
    """One outer step: y update, then CG, then x ← x − α ∇Φ̂_μ. Shapes and finite values."""
    x0 = jnp.asarray(3.0)
    y0 = jnp.asarray(0.5)
    alpha = 0.05
    mu = 0.4
    out = probe(
        x0,
        y0,
        T=1,
        K=20,
        N=4,
        v0=jnp.asarray(0.0),
        alpha=alpha,
        beta=0.1,
        mu=mu,
        r=0.1,
        varrho=1e-3,
        delta_G=1e-4,
        delta_K=5,
        key=jax.random.PRNGKey(2),
    )
    assert out["x"].shape == x0.shape
    assert out["y"].shape == y0.shape
    assert out["v"].shape == y0.shape
    assert out["xs"].shape[0] == 2
    assert out["ys"].shape[0] == 2
    assert out["hypergrads"].shape[0] == 1
    assert jnp.isfinite(out["x"])
    assert jnp.isfinite(out["y"])
    assert jnp.isfinite(out["v"])
    assert jnp.isfinite(out["hypergrad"])
    oracles = fig1_oracles()
    hg = hypergradient(x0, out["y"], out["v"], oracles)
    np.testing.assert_allclose(out["x"], x0 - alpha * hg, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["hypergrad"], hg, rtol=1e-5, atol=1e-5)


def test_probe_escape_from_saddle_vs_r0():
    """T=80 from (3,0) with perturbation yields |y| larger than the r=0 control."""
    x0, y0 = jnp.asarray(3.0), jnp.asarray(0.0)
    common = dict(
        T=80,
        K=40,
        N=4,
        v0=jnp.asarray(0.0),
        alpha=0.05,
        beta=0.1,
        mu=lambda t: 1.0 / ((t + 1) ** 0.5),
        varrho=1e-3,
        delta_G=1e-4,
        delta_K=10,
    )
    out_p = probe(x0, y0, r=0.15, key=jax.random.PRNGKey(3), **common)
    out_c = probe(x0, y0, r=0.0, key=jax.random.PRNGKey(4), **common)
    assert float(jnp.abs(out_p["y"])) > float(jnp.abs(out_c["y"]))
    assert float(jnp.abs(out_p["y"])) > 0.2
    assert float(jnp.abs(out_c["y"])) < 0.05


def test_autodiff_oracles_match_closed_forms():
    """Generic jax.grad/hessian path agrees with Figure 1 closed forms."""
    ad = autodiff_oracles(f, g)
    cl = fig1_oracles()
    x, y = _to_arrays(2.0, 1.0)
    np.testing.assert_allclose(ad.f(x, y), cl.f(x, y), rtol=1e-6)
    np.testing.assert_allclose(ad.g(x, y), cl.g(x, y), rtol=1e-6)
    np.testing.assert_allclose(ad.grad_x_f(x, y), cl.grad_x_f(x, y), rtol=1e-5)
    np.testing.assert_allclose(ad.grad_y_f(x, y), cl.grad_y_f(x, y), rtol=1e-5)
    np.testing.assert_allclose(ad.grad_y_g(x, y), cl.grad_y_g(x, y), rtol=1e-5)
    np.testing.assert_allclose(ad.hess_yy_g(x, y), cl.hess_yy_g(x, y), rtol=1e-5)
    np.testing.assert_allclose(ad.hess_xy_g(x, y), cl.hess_xy_g(x, y), rtol=1e-5)
