"""Metric utilities for constraint RMSE, variability, KLD, and MELR."""

import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax.scipy.integrate import trapezoid
import jax
from functools import partial
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def _single_calculate_constraint_rmse(
    predicted_samples: jnp.ndarray, condition_reference_samples: jnp.ndarray
) -> float:
    """Calculate relative RMSE between predicted and reference samples as defined in the supplementary material (note the choice of the denominator).

    Computes per-sample L2 error divided by the predicted sample L2 norm,
    then averages over the batch.

    Args:
        predicted_samples: Array of predicted samples shaped (N, ...).
        condition_reference_samples: Array of reference samples shaped (N, ...).
    """

    diff_norm = jnp.linalg.norm(predicted_samples - condition_reference_samples, axis=1)
    predicted_norm = jnp.linalg.norm(predicted_samples, axis=1)
    relative_errors = jnp.where(predicted_norm != 0, diff_norm / predicted_norm, 0.0)
    return jnp.mean(relative_errors)


@jax.jit
def calculate_constraint_rmse(
    predicted_samples: jnp.ndarray,
    condition_reference_samples: jnp.ndarray,
    C: jnp.ndarray,
) -> float:
    x = jnp.squeeze(predicted_samples, -1)
    C = C.astype(x.dtype)
    Cx = jnp.einsum("ncd,od->nco", x, C)
    predicted_samples_red_dim = Cx[..., None]
    vec_c = jax.vmap(_single_calculate_constraint_rmse, in_axes=(1, 0), out_axes=0)(
        predicted_samples_red_dim, condition_reference_samples
    )
    return jnp.mean(vec_c)


def _single_calculate_sample_variability(generated_samples: jnp.ndarray) -> float:
    """
    Calculates the sample variability for a set of generated conditional samples.

    This metric is defined in "Debias Coarsely, Sample Conditionally: Statistical
    Downscaling through Optimal Transport and Probabilistic Diffusion Models"
    (Supplementary Material, Appendix C, Eq. 36). It is described as the
    mean pixel-wise standard deviation in the generated conditional samples for a
    given condition.

    The formula is:
    Var = sqrt( (1 / (N * d)) * sum_{n=1 to N} sum_{m=1 to d} (u_nm - u_mean_m)^2 )

    Where:
    - N is the number of samples for a given condition.
    - d is the number of dimensions (pixels) in each sample.
    - u_nm is the value of the m-th dimension of the n-th sample.
    - u_mean_m is the mean value of the m-th dimension across all N samples.

    This can be simplified to the root mean square of the pixel-wise standard
    deviations.

    Args:
        generated_samples (jnp.ndarray): A 2D numpy array of shape (N, d),
            where N is the number of samples and d is the number of features
            or pixels for each sample. All samples should be generated from the
            same condition.

    Returns:
        float: The calculated sample variability, a single non-negative value.
    """
    # Calculate the variance for each pixel/dimension across all samples.
    # The `axis=0` argument computes the variance along the "N" dimension.
    # jnp.var uses N in the denominator by default (ddof=0), which matches the formula.
    pixel_wise_variances = jnp.var(generated_samples, axis=0)

    # Calculate the mean of these variances.
    mean_variance = jnp.mean(pixel_wise_variances)

    # The sample variability is the square root of the mean variance.
    sample_variability = jnp.sqrt(mean_variance)

    return sample_variability


@jax.jit
def calculate_sample_variability(generated_samples: jnp.ndarray) -> float:
    vec_c = jax.vmap(_single_calculate_sample_variability, in_axes=(1,), out_axes=0)(
        generated_samples
    )
    return jnp.mean(vec_c)


def _single_dimension_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculates the Kullback-Leibler divergence for a single dimension.
    """

    # Extract the 1D marginal data for the current dimension
    pred_data = jnp.squeeze(predicted_samples)
    ref_data = jnp.squeeze(reference_samples)

    # Create Kernel Density Estimations for both distributions
    kde_pred = gaussian_kde(pred_data, bw_method="scott")
    kde_ref = gaussian_kde(ref_data, bw_method="scott")

    # Create a common grid of points to evaluate the PDFs
    min_val = jnp.minimum(jnp.min(pred_data), jnp.min(ref_data))
    max_val = jnp.maximum(jnp.max(pred_data), jnp.max(ref_data))
    max_val = jnp.where(max_val == min_val, min_val + 1e-6, max_val)
    grid = jnp.linspace(min_val, max_val, 256)

    # Evaluate the PDFs on the grid
    pdf_pred = kde_pred(grid)
    pdf_ref = kde_ref(grid)

    # Calculate the integrand for KL divergence without boolean indexing
    mask = pdf_ref > epsilon
    integrand = jnp.where(mask, pdf_ref * jnp.log(pdf_ref / (pdf_pred + epsilon)), 0.0)

    # Approximate the integral using the trapezoidal rule over full grid
    kld_m = trapezoid(integrand, x=grid)

    return kld_m


@jax.jit
def _single_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculates the aggregated Kullback-Leibler divergence using Kernel Density Estimation.

    This metric is defined in "Debias Coarsely, Sample Conditionally: Statistical
    Downscaling through Optimal Transport and Probabilistic Diffusion Models"
    (Supplementary Material, Appendix C, Eq. 35). It computes the KL divergence
    for the 1D marginal distributions of each dimension and sums them up.

    The formula is:
    KLD = sum_{m=1 to d} integral( p_m_ref(v) * log(p_m_ref(v) / p_m_pred(v)) dv )

    Where:
    - d is the number of dimensions (pixels).
    - p_m_ref and p_m_pred are the 1D marginal probability density functions
      (PDFs) for the m-th dimension, estimated using KDE with Scott's rule
      for bandwidth selection.
    - The integral is approximated using the trapezoidal rule.

    Args:
        predicted_samples (np.ndarray): A 2D numpy array of shape (N, d),
            representing the generated samples.
        reference_samples (np.ndarray): A 2D numpy array of shape (M, d),
            representing the ground truth or reference samples HFHR. N can be different from M.
        epsilon (float): A small value to add to the predicted PDF to avoid
            division by zero in the log.

    Returns:
        float: The total KLD over all dimensions.
    """
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )

    kld_vec = jax.vmap(
        _single_dimension_calculate_kld, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, epsilon)
    total_kld = jnp.sum(kld_vec)

    return total_kld


@jax.jit
def calculate_kld_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )
    kld = _single_calculate_kld(pooled_predicted_samples, reference_samples, epsilon)
    return kld


@partial(jax.jit, static_argnames="sample_shape")
def _get_k_grids(sample_shape: tuple):
    """Creates JAX-compatible wavenumber grids for a given static shape."""
    freqs = [jnp.fft.fftfreq(n, d=1.0 / n) for n in sample_shape]
    k_grids = jnp.meshgrid(*freqs, indexing="ij")
    k_magnitude = jnp.sqrt(sum(k**2 for k in k_grids))
    # Use a fixed, large number of bins to make it JIT-compatible.
    max_bins = max(sample_shape) // 2 + 1
    k_bins = jnp.arange(0.5, max_bins)

    counts, _ = jnp.histogram(k_magnitude.flatten(), bins=k_bins)

    return k_magnitude, k_bins, counts


def _get_energy_spectrum_for_one_sample(
    sample: jnp.ndarray,
    sample_shape: tuple,
    k_magnitude: jnp.ndarray,
    k_bins: jnp.ndarray,
) -> jnp.ndarray:
    """Helper function to compute the energy spectrum for a single flattened sample."""
    sample_reshaped = sample.reshape(sample_shape)

    fft_coeffs = jnp.fft.fftn(sample_reshaped)
    power_spectrum = jnp.abs(fft_coeffs) ** 2

    energy_spectrum, _ = jnp.histogram(
        k_magnitude.flatten(), bins=k_bins, weights=power_spectrum.flatten()
    )

    return energy_spectrum


@partial(jax.jit, static_argnames=["sample_shape", "weighted"])
def calculate_melr_pooled(
    predicted_samples: jnp.ndarray,  # Shape: (N, C, D, 1) e.g., (10, 2, 192, 1)
    reference_samples: jnp.ndarray,  # Shape: (M, D, 1) e.g., (163840, 192, 1)
    sample_shape: tuple,
    weighted: bool,
    epsilon: float = 1e-10,
) -> jnp.ndarray:
    """
    Calculates average MELR by vmapping over the C-axis (conditions).
    """
    if predicted_samples.ndim == 3:
        predicted_samples = predicted_samples[:, None, :, :]
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    pred_clean = jnp.squeeze(pooled_predicted_samples, axis=-1)  # Shape: (N*C, D)
    ref_clean = jnp.squeeze(reference_samples, axis=-1)  # Shape: (M, D)

    k_magnitude, k_bins, counts = _get_k_grids(sample_shape)

    vmapped_spectrum_fn = jax.vmap(
        _get_energy_spectrum_for_one_sample, in_axes=(0, None, None, None)
    )
    E_pred_batch = vmapped_spectrum_fn(pred_clean, sample_shape, k_magnitude, k_bins)
    E_ref_batch = vmapped_spectrum_fn(ref_clean, sample_shape, k_magnitude, k_bins)

    E_pred = jnp.mean(E_pred_batch, axis=0)
    E_ref = jnp.mean(E_ref_batch, axis=0)

    if E_pred.shape[0] != E_ref.shape[0]:
        raise ValueError(
            f"Energy spectrum shapes do not match: E_pred={E_pred.shape}, E_ref={E_ref.shape}"
        )

    log_ratios = jnp.abs(jnp.log((E_pred + epsilon) / (E_ref + epsilon)))

    def weighted_calc():
        weights = E_ref / jnp.sum(E_ref)
        return jnp.sum(weights * log_ratios)

    def unweighted_calc():
        return jnp.mean(log_ratios)

    return jax.lax.cond(weighted, weighted_calc, unweighted_calc)


def _single_dimension_calculate_wass1(
    predicted_samples_1d: jnp.ndarray,
    reference_samples_1d: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    Calculates the Wasserstein-1 metric for a single dimension using empirical CDFs.

    The integral is approximated over the fixed range [-20, 20] as specified.
    """
    # Define the integration range and bins
    integration_range = [
        -20.0,
        20.0,
    ]  # this can be changed to the range of the data!!! I think also sufficient for KS
    bins = jnp.linspace(integration_range[0], integration_range[1], num_bins + 1)

    # Ensure data is 1D for histogram
    pred_data = jnp.squeeze(predicted_samples_1d)
    ref_data = jnp.squeeze(reference_samples_1d)

    # Compute histograms (empirical PDFs)
    counts_pred, _ = jnp.histogram(pred_data, bins=bins, range=integration_range)
    counts_ref, _ = jnp.histogram(ref_data, bins=bins, range=integration_range)

    # Compute empirical CDFs. Add epsilon for numerical stability if sum is zero.
    total_pred = jnp.sum(counts_pred)
    total_ref = jnp.sum(counts_ref)

    cdf_pred = jnp.cumsum(counts_pred) / (total_pred + 1e-10)
    cdf_ref = jnp.cumsum(counts_ref) / (total_ref + 1e-10)

    # Calculate the absolute difference between CDFs
    cdf_diff = jnp.abs(cdf_pred - cdf_ref)

    # Get bin centers for trapezoidal integration
    # The CDF values correspond to the probability mass *within* each bin
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Approximate the integral using the trapezoidal rule
    # This is equivalent to sum(|CDF_pred - CDF_ref| * dz)
    wass1_m = trapezoid(cdf_diff, x=bin_centers)

    return wass1_m


@partial(jax.jit, static_argnames="num_bins")
def _single_calculate_wass1(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    Calculates the aggregated Wasserstein-1 metric (Wass1) as defined in Eq. 38.

    It computes the Wass1 metric for the 1D marginal distributions of each
    dimension and averages them.

    The formula is:
    Wass1 = (1/d) * sum_{m=1 to d} integral( |CDF_pred,m(z) - CDF_ref,m(z)| dz )

    Args:
        predicted_samples (jnp.ndarray): Array of shape (N, d, 1),
            representing the generated samples.
        reference_samples (jnp.ndarray): Array of shape (M, d, 1),
            representing the reference samples. N can be different from M.
        num_bins (int): The number of bins to use for the empirical CDF
            and integration. Must be static for JIT.

    Returns:
        float: The mean Wass1 metric over all dimensions.
    """
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )
    if predicted_samples.shape[2] != 1 or reference_samples.shape[2] != 1:
        raise ValueError(
            f"Expected trailing dimension of 1, but got {predicted_samples.shape} and {reference_samples.shape}"
        )

    # Vectorize the 1D calculation over all dimensions (axis=1)
    # in_axes=(1, 1, None) maps over dimension 'd' for both samples
    # and passes the static 'num_bins' argument
    wass1_vec = jax.vmap(
        _single_dimension_calculate_wass1, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, num_bins)

    # Average the Wass1 metric across all dimensions
    mean_wass1 = jnp.mean(wass1_vec)

    return mean_wass1


@partial(jax.jit, static_argnames="num_bins")
def calculate_wass1_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """
    JIT-compiled entry point for pooled Wass1 calculation.

    Pools the batch and condition axes of predicted_samples before metric calculation.

    Args:
        predicted_samples (jnp.ndarray): Shape (N, C, D, 1)
        reference_samples (jnp.ndarray): Shape (M, D, 1)
        num_bins (int): Number of bins for histogram.

    Returns:
        float: The mean Wass1 metric.
    """
    # Pool the N (batch) and C (condition) axes
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    wass1 = _single_calculate_wass1(
        pooled_predicted_samples, reference_samples, num_bins
    )
    return wass1
