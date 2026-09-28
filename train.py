"""CPU demo of PROBE on the Figure 1 scalar toy (arXiv:2609.30501)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from probe_blo.probe import probe
from probe_blo.toy import f, g, grad_y_g, hess_yy_g


def _summarize(name, out):
    x = out["x"]
    y = out["y"]
    gx = float(jnp.abs(grad_y_g(x, y)))
    lam = float(hess_yy_g(x, y))
    print(f"=== {name} ===")
    print(f"final (x, y) = ({float(x):.6f}, {float(y):.6f})")
    print(f"f = {float(f(x, y)):.6f}")
    print(f"g = {float(g(x, y)):.6f}")
    print(f"|grad_y g| = {gx:.6e}")
    print(f"lambda_min = {lam:.6f}")
    xs = np_traj(out["xs"])
    ys = np_traj(out["ys"])
    n = len(xs)
    idxs = sorted(set([0, n // 2, n - 1]))
    print("trajectory (t, x, y):")
    for i in idxs:
        print(f"  t={i:4d}  x={xs[i]: .6f}  y={ys[i]: .6f}")
    print()


def np_traj(a):
    return [float(z) for z in a]


def main():
    common = dict(
        T=200,
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

    out_02 = probe(
        jnp.asarray(0.0),
        jnp.asarray(2.0),
        r=0.15,
        key=jax.random.PRNGKey(0),
        **common,
    )
    _summarize("PROBE from (0, 2)", out_02)

    out_30 = probe(
        jnp.asarray(3.0),
        jnp.asarray(0.0),
        r=0.15,
        key=jax.random.PRNGKey(1),
        **common,
    )
    _summarize("PROBE from (3, 0)", out_30)

    out_ctrl = probe(
        jnp.asarray(3.0),
        jnp.asarray(0.0),
        r=0.0,
        key=jax.random.PRNGKey(2),
        **common,
    )
    _summarize("r=0 control from (3, 0)", out_ctrl)

    y_probe = abs(float(out_30["y"]))
    y_ctrl = abs(float(out_ctrl["y"]))
    print("saddle comparison from (3, 0):")
    print(f"  PROBE |y| = {y_probe:.6f}  (r>0 leaves the saddle)")
    print(f"  control |y| = {y_ctrl:.6f}  (r=0 stays near y=0)")
    print(f"  control stayed near y=0: {y_ctrl < 0.05}")
    print(f"  PROBE left the saddle: {y_probe > 0.2}")


if __name__ == "__main__":
    main()
