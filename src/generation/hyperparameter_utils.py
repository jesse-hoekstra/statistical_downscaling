"""Utilities to score WAN guidance hyperparameters.

This module provides a single helper `hyperparameter_step` that:
- runs a short conditional sampling sweep using the current settings,
- computes MELR (unweighted) and sample variability for the produced samples,
- returns those two scalars so the caller can populate tuning grids/plots.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.utils_metrics import (
    calculate_sample_variability,
    calculate_melr_pooled,
)
from src.generation.sampler_utils import sample_wan_guided


def hyperparameter_step(
    run_sett, diffusion_scheme, denoise_fn, y, key_wan, u_hfhr_samples
):
    """Run a small WAN-conditional sampling and compute tuning metrics.

    Args:
        run_sett: Nested settings dict, with the current WAN guidance settings values for the current hyperparameter tuning experiment.
        diffusion_scheme: Diffusion scheme instance created for the data std.
        denoise_fn: Callable denoiser used by the sampler.
        y: Conditioning array with shape (num_conditions, d'), typically taken
           from the LR observations. Only the first few are used here.
        key_wan: JAX PRNGKey for sampling.
        u_hfhr_samples: Reference HF samples array for dataset-level metrics.

    Returns:
        Tuple (melr_unweighted, sample_variability), both Python floats/JAX scalars.
    """

    samples = sample_wan_guided(
        diffusion_scheme,
        denoise_fn,
        y_bar=y[:5],
        rng_key=key_wan,
        num_samples=10,
        run_sett=run_sett,
    )

    melr_unw_val = calculate_melr_pooled(
        samples,
        u_hfhr_samples,
        sample_shape=(run_sett["global"]["d"],),
        weighted=False,
        epsilon=float(run_sett["metrics"]["epsilon"]),
    )
    sam_var_val = calculate_sample_variability(samples)

    return melr_unw_val, sam_var_val
