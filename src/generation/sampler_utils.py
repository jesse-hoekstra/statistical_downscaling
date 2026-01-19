"""Sampling utilities for unconditional, linear-guided, and PDE-guided draws.

Wraps common sampler configurations used in experiments: plain diffusion
sampling, WAN-style linear constraints, and PDE-guided sampling via custom
drift updates.
"""

import jax
import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib

from src.generation.swirl_dynamics_new_guidance.guidance import LinearConstraint
from src.generation.swirl_dynamics_new_sampler.samplers import NewDriftSdeSampler
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def sample_unconditional(
    diffusion_scheme, denoise_fn, rng_key: jax.Array, num_samples: int
):
    """Generate unconditional samples using an SDE sampler.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        rng_key: JAX PRNG key for sampling.
        num_samples: Number of independent samples to draw.

    Returns:
        Array of generated samples with shape `(num_samples, d, 1)`.
    """
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=run_sett["exp_tspan"]["num_steps"],
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
    return generate(rng=rng_key, num_samples=num_samples)


def less_memory_sample_wan_guided(
    diffusion_scheme,
    denoise_fn,
    y_bar: jnp.ndarray,
    rng_key: jax.Array,
    num_samples: int,
):
    """WAN-style guided sampling with fixed chunking for lower memory.

    Example: if `y_bar` has shape `(512, 48, 1)` and `splits=16`, this will
    split into 16 chunks of `(32, 48, 1)`. For `num_samples=128`, each chunk
    yields `(128, 32, d, 1)` (with `d` from settings), and the concatenated
    result is `(128, 512, d, 1)`.

    Returns `(num_samples, C, d, 1)` with `C=y_bar.shape[0]`.
    """
    splits = 16
    total_conditions = int(y_bar.shape[0])
    if total_conditions % splits != 0:
        raise ValueError(
            f"total conditions {total_conditions} must be divisible by splits {splits}"
        )
    batch_size = total_conditions // splits

    # Reshape to (splits, batch_size, ...), preserving order
    new_shape = (splits, batch_size) + tuple(y_bar.shape[1:])
    y_bar_in_batches = jnp.reshape(y_bar, new_shape)

    # Keys per split; vmap over both chunks and keys
    keys = jax.random.split(rng_key, splits)

    @jax.jit
    def run_one(chunk, key):
        return sample_wan_guided(
            diffusion_scheme,
            denoise_fn,
            y_bar=chunk,
            rng_key=key,
            num_samples=num_samples,
        )  # (N, batch_size, d, 1)

    all_chunks = []
    for i in range(
        splits
    ):  # cannot use vmap as double vmap is problem when running on server
        chunk_result = run_one(y_bar_in_batches[i], keys[i])
        all_chunks.append(chunk_result)

    # Stack all results from the list
    per_chunk = jnp.stack(all_chunks, axis=0)  # (splits, N, B, d, 1)

    # Reassemble across condition chunks into (N, C, d, 1)
    per_chunk = jnp.transpose(per_chunk, (1, 0, 2, 3, 4))  # (N, splits, B, d, 1)
    combined = jnp.reshape(
        per_chunk, (num_samples, total_conditions) + tuple(per_chunk.shape[3:])
    )
    return combined


def sample_wan_guided(
    diffusion_scheme,
    denoise_fn,
    y_bar: jnp.ndarray,
    rng_key: jax.Array,
    num_samples: int,
):
    """WAN-style guided sampling for a batch of conditions.

    Builds a binary downsampling matrix `C_prime` and applies a linear constraint of Wan er al. (2023). Given a batch of `C`
    conditions in `y_bar`, this function generates `num_samples` independent draws, resulting
    in an output of shape `(N, C, d, 1)`.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        y_bar: Target low-resolution conditions; shape `(C, d')`.
        rng_key: JAX PRNG key for sampling.
        num_samples: Number of independent samples to draw `(N)`.
    Returns:
        Array with shape `(N, C, d, 1)`, where the first axis indexes
        independent draws (each with a different RNG) and the second axis
        indexes the `C` conditions in `y_bar`.
    """

    downsampling_factor = int(run_sett["general"]["d"]) // int(
        run_sett["general"]["d_prime"]
    )
    C_prime = jnp.array(
        [
            [
                1 if j == downsampling_factor * i else 0
                for j in range(int(run_sett["general"]["d"]))
            ]
            for i in range(int(run_sett["general"]["d_prime"]))
        ]
    )
    guidance_transform = LinearConstraint.create(
        C_prime=C_prime,
        y_bar=y_bar,
        norm_guide_strength=run_sett["general"]["norm_guide_strength"],
    )
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(guidance_transform,),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
    inner_num = int(y_bar.shape[0])  # should be same dimensions of conditions in y_bar
    keys = jax.random.split(rng_key, num_samples)
    batched_generate = jax.vmap(lambda k: generate(rng=k, num_samples=inner_num))
    return batched_generate(keys)


def sample_pde_guided(
    diffusion_scheme,
    denoise_fn,
    pde_solver,
    rng_key: jax.Array,
    samples_per_condition: int,
    y: jnp.ndarray,
):
    """Generate samples guided by our additional gradient term.

    Uses a custom sampler with drift adjustments provided by
    `pde_solver.grad_log_h_batched_one_per_model`.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        pde_solver: Solver exposing `grad_log_h_batched_one_per_model` and
            attribute `num_models`.
        rng_key: JAX PRNG key for sampling.
        samples_per_condition: Number of batches; each batch generates
            `pde_solver.num_models` samples.
        y: Target low-resolution conditions; shape `(M, d')` or `(M, d', 1)`
           where `M == pde_solver.num_models`.

    Returns:
        Array with shape `(samples_per_condition, M, d, 1)`.
    """
    num_models = int(pde_solver.num_models)
    if y.shape[0] != num_models:
        raise ValueError(
            f"`y` must have leading size {num_models}, but got {y.shape[0]}"
        )
    sampler = NewDriftSdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        guidance_fn=pde_solver.grad_log_h_batched_one_per_model,
        apply_denoise_at_end=True,
        return_full_paths=False,
    )

    # Generate `samples_per_condition` independent batches; reuse the same `y` batch each time.
    keys = jax.random.split(rng_key, samples_per_condition)
    generate_one = jax.jit(
        lambda k: sampler.generate(
            rng=k, num_samples=num_models, guidance_inputs={"y": y}
        )
    )

    # scan requires a carry argument; use None
    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all
