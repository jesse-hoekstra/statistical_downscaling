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

    samples = sample_wan_guided(
        diffusion_scheme,
        denoise_fn,
        y_bar=y[:50],
        rng_key=key_wan,
        num_samples=50,
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
