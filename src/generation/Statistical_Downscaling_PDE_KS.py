"""KS statistical downscaling PDE built on a DGM and diffusion schedule.

Defines the PDE residual and terminal losses used for conditional sampling with
linear observations Cx = y. The solver uses a Deep Galerkin Network (DGM) for
V(t, x) and derives drift/diffusion terms from a provided diffusion scheme and
denoiser.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from src.generation.PDE_solver import PDE_solver
from swirl_dynamics.lib.diffusion import diffusion
from typing import Any, TypeAlias
from collections.abc import Mapping

Array: TypeAlias = jax.Array
ArrayMapping: TypeAlias = Mapping[str, Array]
Params: TypeAlias = Mapping[str, Any]


def _dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    return jax.grad(lambda t: jnp.log(f(t)))


def _dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return jax.grad(lambda t: jnp.square(f(t)))


def _build_C_prime(d: int, d_prime: int) -> jax.Array:
    downsampling_factor = d // d_prime

    return jnp.array(
        [
            [1 if j == downsampling_factor * i else 0 for j in range(d)]
            for i in range(d_prime)
        ]
    ).astype(jnp.float32)


class KSStatisticalDownscalingPDESolver(PDE_solver):
    """PDE for statistical downscaling with linear observation Cx = y.

    Implements interior-residual and terminal losses using a DGM for V(t, x)
    and drift/diffusion terms derived from a diffusion scheme and denoiser.
    """

    def __init__(
        self,
        samples: np.ndarray,
        settings: dict,
        denoise_fn,
        scheme,
        rng_key: jax.Array | None = None,
    ):
        """Initialize the statistical downscaling PDE solver.

        Args:
            samples: Training samples (may be used by downstream utilities).
            y_bar: Conditioning observations at low resolution; shape `(M, d')`
                or `(d',)`.
            settings: Hierarchical settings dictionary for base solver and DGM.
            denoise_fn: Callable denoiser used in drift construction.
            scheme: Diffusion scheme providing `sigma`, `scale`, and their
                derivatives via autodiff.
            rng_key: Optional PRNG key for deterministic initialization.
        """
        super().__init__(settings=settings, rng_key=rng_key)
        self.C_prime = _build_C_prime(self.d, self.d_prime)
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.lambda_value = jnp.float32(self.run_sett_pde_solver["lambda"])

    @partial(jax.jit, static_argnums=0)
    def loss_fn(
        self,
        params,
        t_interior,
        x_interior,
        y_interior,
        t_terminal,
        x_terminal,
        y_terminal,
    ):
        """Compute PDE residual and terminal losses for downscaling.

        Args:
            params: DGM parameters.
            t_interior: Interior times `(B, 1)`.
            x_interior: Interior states `(B, d)`.
            y_interior: Interior targets `(B, d)`.
            t_terminal: Terminal times `(B, 1)` (typically all T).
            x_terminal: Terminal states `(B, d)`.
            y_terminal: Terminal targets `(B, d)`.

        Returns:
            Tuple `(L1, L3, total)` where `L1` is interior residual MSE and
            `L3` enforces the terminal constraint via a smooth kernel.
        """

        def V_single(ts: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
            return self.net.apply(params, ts[None], xs[None], ys[None]).squeeze()

        dV_dt_fn = jax.grad(lambda ts, xs, ys: V_single(ts, xs, ys))
        dV_dx_fn = jax.grad(lambda xs, ts, ys: V_single(ts, xs, ys))
        H_fn = jax.hessian(lambda xs, ts, ys: V_single(ts, xs, ys))

        V_t = jax.vmap(lambda ts, xs, ys: dV_dt_fn(ts.squeeze(), xs, ys))(
            t_interior,
            x_interior,
            y_interior,
        )
        V_x = jax.vmap(lambda ts, xs, ys: dV_dx_fn(xs, ts.squeeze(), ys))(
            t_interior,
            x_interior,
            y_interior,
        )
        V_xx = jax.vmap(lambda ts, xs, ys: H_fn(xs, ts.squeeze(), ys))(
            t_interior, x_interior, y_interior
        )

        trace_term = jax.vmap(lambda m: jnp.trace(m))(V_xx)

        def _drift(x: Array, t: Array) -> Array:
            x = x[None, ..., None]
            assert not t.ndim, "`t` must be a scalar."
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = jnp.divide(x, s)
            dlog_sigma_dt = _dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = _dlog_dt(self.scheme.scale)(t)
            drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
            drift -= 2 * dlog_sigma_dt * s * self.denoise_fn(x_hat, sigma)

            return jnp.squeeze(drift)

        def _half_diffusion2(x: Array, t: Array) -> Array:
            assert not t.ndim, "`t` must be a scalar."
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            dsquare_sigma_dt = _dsquare_dt(self.scheme.sigma)(t)
            return (1 / 2) * (s**2) * dsquare_sigma_dt

        drifts = jax.vmap(_drift)(x_interior, jnp.squeeze(t_interior, -1))
        diffusions = jax.vmap(_half_diffusion2)(x_interior, jnp.squeeze(t_interior, -1))

        grad_norm_sq = jnp.sum(jnp.square(V_x), axis=-1)

        PDE_residual = (
            V_t.reshape(-1)
            + jnp.einsum("bd,bd->b", drifts, V_x)
            + diffusions * (trace_term + grad_norm_sq)
        ).reshape(-1, 1)
        L1 = jnp.mean(jnp.square(PDE_residual))

        V_term = self.net.apply(params, t_terminal, x_terminal, y_terminal)
        Cx = x_terminal @ self.C_prime.T

        diff = Cx - y_terminal
        sqnorm = jnp.sum(jnp.square(diff), axis=-1)
        target = -jnp.log(self.lambda_value) - (sqnorm / (self.lambda_value**2))
        target = target.reshape(-1, 1)
        L3 = jnp.mean(jnp.square(V_term - target))

        aux = {
            "L1_loss": L1,
            "L3_loss": L3,
            "total_loss": (L1 + L3),
        }
        loss_val = L1 + L3
        return loss_val, aux
