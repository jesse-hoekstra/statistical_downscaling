"""PDE-based statistical downscaling for the 1D Kuramoto–Sivashinsky setting.

This module defines a solver that learns a value function V(t, x, y) as an approximation to the solution of the PDE  used to
derive guidance for conditional diffusion sampling with linear observations
 C' x = y'. The solver implements:
- construction of the observation matrix C' (mask),
- optional per-dimension normalization from training samples,
- a loss composed of a PDE interior residual (L1) and a terminal loss (L3),
- utilities to compute gradients of log h(t, x, y) for sampler guidance.
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
    )


class KSStatisticalDownscalingPDESolver(PDE_solver):
    """PDE solver specialized for KS downscaling with linear observations.

    The class ties together a diffusion scheme, a denoiser (for drift terms),
    and a Deep Galerkin Network (from the base PDE_solver) to optimize a value
    function that enforces both a PDE residual in the interior and a terminal
    constraint consistent with C' x = y'.
    """

    def __init__(
        self,
        samples: np.ndarray,
        settings: dict,
        denoise_fn,
        scheme,
    ):
        """Initialize the solver and optional normalization statistics.

        Args:
            samples: Training HF samples of shape (N, d, 1). Used only to
                estimate normalization statistics when `normalize_data=True`.
            settings: Nested configuration passed to the base solver.
            denoise_fn: Callable denoiser used to build SDE drift terms.
            scheme: Diffusion scheme providing `sigma`, `scale`, and their
                derivatives through autodiff.
        """
        super().__init__(settings=settings)
        self.C_prime = _build_C_prime(self.d, self.d_prime)
        self.scheme = scheme
        self.denoise_fn = denoise_fn
        self.lambda_value = float(self.run_sett_pde_solver["lambda"])
        self.normalize_data = bool(self.run_sett_pde_solver["normalize_data"])
        if self.normalize_data:
            self.x_mean, self.x_std, self.y_mean, self.y_std = self._calibrate_data(
                samples
            )
        else:
            self.x_mean = jnp.zeros(self.d)
            self.x_std = jnp.ones(self.d)
            self.y_mean = jnp.zeros(self.d_prime)
            self.y_std = jnp.ones(self.d_prime)

    def _calibrate_data(self, samples):
        """Compute per-dimension mean/std for x and induced y = C' x."""
        x_mean = jnp.mean(samples.squeeze(-1), axis=0)
        x_std = jnp.std(samples.squeeze(-1), axis=0) + 1e-6

        samples_y = jnp.matmul(samples.squeeze(-1), self.C_prime.T)
        y_mean = jnp.mean(samples_y, axis=0)
        y_std = jnp.std(samples_y, axis=0) + 1e-6
        return x_mean, x_std, y_mean, y_std

    def _normalize_data(self, x, y):
        """Apply stored normalization to x and y."""
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def _denormalize_data(self, x_norm, y_norm):
        """Invert normalization for x and y."""
        return x_norm * self.x_std + self.x_mean, y_norm * self.y_std + self.y_mean

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
            t_terminal: Terminal times `(B, 1)` .
            x_terminal: Terminal states `(B, d)`.
            y_terminal: Terminal targets `(B, d)`.

        Returns:
            Tuple `(L1, L3, total)` where `L1` is interior residual MSE and
            `L3` enforces the terminal constraint via a smooth kernel.
        """

        x_interior_norm, y_interior_norm = self._normalize_data(x_interior, y_interior)

        def V_single(ts: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
            return self.net.apply(params, ts[None], xs[None], ys[None]).squeeze()

        dV_dt_fn = jax.grad(lambda ts, xs, ys: V_single(ts, xs, ys))
        dV_dx_fn = jax.grad(lambda xs, ts, ys: V_single(ts, xs, ys))

        V_t = jax.vmap(lambda ts, xs, ys: dV_dt_fn(ts.squeeze(), xs, ys))(
            t_interior,
            x_interior_norm,
            y_interior_norm,
        )
        V_x_norm = jax.vmap(lambda ts, xs, ys: dV_dx_fn(xs, ts.squeeze(), ys))(
            t_interior,
            x_interior_norm,
            y_interior_norm,
        )

        inv_sigma_sq = 1 / self.x_std**2

        def _laplacian_exact(ts, xs, ys, inv_sigma_sq):
            def f(x):
                return V_single(ts, x, ys)

            grad_f = jax.grad(f)
            eye = jnp.eye(xs.shape[-1], dtype=xs.dtype)

            def diag_second(ei):
                _, hv = jax.jvp(grad_f, (xs,), (ei,))
                return jnp.vdot(ei, hv) * jnp.vdot(ei, inv_sigma_sq)

            return jnp.sum(jax.vmap(diag_second)(eye))

        trace_term = jax.vmap(
            lambda ts, xs, ys: _laplacian_exact(ts.squeeze(), xs, ys, inv_sigma_sq)
        )(t_interior, x_interior_norm, y_interior_norm)
        V_x = V_x_norm / self.x_std

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

        x_terminal_norm, y_terminal_norm = self._normalize_data(x_terminal, y_terminal)

        V_term = self.net.apply(params, t_terminal, x_terminal_norm, y_terminal_norm)

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
            "log_L1": jnp.log(L1 + 1e-6),
            "log_L3": jnp.log(L3 + 1e-6),
            "log_total": jnp.log(L1 + L3 + 1e-6),
        }
        loss_val = L1 + L3
        return loss_val, aux
