"""Modules for guidance transforms for denoising functions."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import chex
import flax
import jax
import jax.numpy as jnp

Array = jax.Array
PyTree = Any
ArrayMapping = Mapping[str, Array]
DenoiseFn = Callable[[Array, Array, ArrayMapping | None], Array]


class Transform(Protocol):
    """Transforms a denoising function to follow some guidance.

    One may think of these transforms as instances of Python decorators,
    specifically made for denoising functions. Each transform takes a base
    denoising function and extends it (often using some additional data) to build
    a new denoising function with the same interface.
    """

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
    ) -> DenoiseFn:
        """Constructs a guided denoising function.

        Args:
          denoise_fn: The base denoising function.
          guidance_inputs: A dictionary containing inputs used to construct the
            guided denoising function. Note that all transforms *share the same
            input dict*, therefore all transforms should use different fields from
            this dict (unless absolutely intended) to avoid potential name clashes.

        Returns:
          The guided denoising function.
        """
        ...


@flax.struct.dataclass
class LinearConstraint:
    """Guidance via gradient correction followed by projection.

     This class implements the precise constrained sampling method from Wan et al.
     (2023), following the formula:
       D̃ = (C')†y' + (I - VVᵀ)[D(x̂_t,σ_t) - α * ∇_{x̂_t}||C'D(x̂_t,σ_t) - y'||²]
     where α is the guidance strength.

     The process involves first correcting the denoiser's output with a gradient
     step and then projecting the result onto the constraint manifold.

    Attributes:
       C_prime: The constraint matrix `C'` itself.
       y_bar: The observation vector `y'`.
       v_matrix: The pre-computed `V` matrix from the SVD of `C'`.
       v_transpose: The pre-computed `Vᵀ` matrix from the SVD of `C'`.
       pseudo_inverse_C_prime: The pre-computed pseudo-inverse `(C')†`.
       guide_strength: The rescaled guidance strength `α`.
    """

    C_prime: Array
    y_bar: Array
    v_matrix: Array
    v_transpose: Array
    pseudo_inverse_C_prime: Array
    guide_strength: chex.Numeric

    @classmethod
    def create(
        cls,
        C_prime: Array,
        y_bar: Array,
        norm_guide_strength: chex.Numeric = 1.0,
    ) -> "LinearConstraint":
        """
        Factory method to create the guidance transform.
        Performs all expensive one-time calculations like SVD.
        """

        # --- Perform expensive calculations ONCE here ---
        num_total_dims = C_prime.shape[1]
        num_constrained_dims = C_prime.shape[0]
        cond_fraction = num_constrained_dims / num_total_dims

        # Rescale guidance strength
        guide_strength = norm_guide_strength
        if cond_fraction > 0:
            guide_strength *= cond_fraction

        _, _, v_transpose = jnp.linalg.svd(C_prime, full_matrices=False)
        v_matrix = v_transpose.T
        pseudo_inverse_C_prime = jnp.linalg.pinv(C_prime)

        return cls(
            C_prime=C_prime,
            y_bar=y_bar,
            v_matrix=v_matrix,
            v_transpose=v_transpose,
            pseudo_inverse_C_prime=pseudo_inverse_C_prime,
            guide_strength=guide_strength,
        )

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping | None = None
    ) -> DenoiseFn:
        """Constructs and returns a guided denoiser with the same interface as `denoise_fn`.

        Behavior:
        - Builds a closure that, when called, applies the linear-constraint guidance as explained before.
        - The returned function can be used transparently by samplers in place of
          the original `denoise_fn`.

        Expected shapes:
        - x̂: Array with leading batch dimension. In this project it is typically
          shape (num_conditions, d, 1), where d is the high-resolution spatial size.
        - sigma: Scalar Array or an Array broadcastable to the batch; the same σ is
          used for all items in the batch during a single call.
        - cond: Optional mapping of conditioning tensors; must be broadcastable to
          the same leading batch size as x̂.
        - y_bar (stored in `self`): shape (num_conditions, d'), one per item in the batch.

        Returns:
        - A callable `_guided_denoise(x, sigma, cond)` that outputs an Array with
          the same shape as `x`, i.e., (num_conditions, d, 1).
        """

        def _guided_denoise(
            x: Array, sigma: Array, cond: ArrayMapping | None = None
        ) -> Array:
            """Applies the modification as explained in Wan et al. (2023)."""
            original_shape = x.shape

            # Compute per-sample gradients directly (no loss returned), with batched y_bar
            def sample_loss(xi: Array, yi: Array) -> Array:
                """Computes the loss for a single sample."""
                den_sample = denoise_fn(xi[None, ...], sigma, cond)[0]
                den_flat = den_sample.reshape(-1)
                y_flat = yi.reshape(-1)
                proj = self.C_prime @ den_flat
                resid = proj - y_flat
                return jnp.sum(resid**2)

            grads_per_sample = jax.vmap(jax.grad(sample_loss))(x, self.y_bar)
            denoised_uncons = denoise_fn(x, sigma, cond)
            denoised_flat = denoised_uncons.reshape(x.shape[0], -1)

            corrected_denoised_flat = (
                denoised_flat
                - self.guide_strength * grads_per_sample.reshape(x.shape[0], -1)
            )

            v_t_dot_corr = jnp.einsum(
                "ij,bj->bi", self.v_transpose, corrected_denoised_flat
            )
            projection = jnp.einsum("ij,bj->bi", self.v_matrix, v_t_dot_corr)
            projected_term = corrected_denoised_flat - projection

            y_bar_flat = self.y_bar.reshape(x.shape[0], -1)
            constraint_comp = jnp.einsum(
                "ij,bj->bi", self.pseudo_inverse_C_prime, y_bar_flat
            )
            final_denoised_flat = projected_term + constraint_comp

            return final_denoised_flat.reshape(original_shape)

        return _guided_denoise
