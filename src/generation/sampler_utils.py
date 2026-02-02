import jax
import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib

from src.generation.swirl_dynamics_new_guidance.guidance import LinearConstraint
from src.generation.swirl_dynamics_new_sampler.samplers import NewDriftSdeSampler


def _build_C_prime(d: int, d_prime: int) -> jax.Array:
    downsampling_factor = d // d_prime

    return jnp.array(
        [
            [1 if j == downsampling_factor * i else 0 for j in range(d)]
            for i in range(d_prime)
        ]
    )


def sample_unconditional(
    diffusion_scheme,
    denoise_fn,
    rng_key: jax.Array,
    num_samples: int,
    num_plots: int,
    run_sett,
):
    """Generate unconditional samples using an SDE sampler.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        rng_key: JAX PRNG key for sampling.
        num_samples: Number of independent samples to draw.
        num_plots: Number of plots to generate, equal to the number of conditions in the conditional samplers.
    Returns:
        Array of generated samples with shape `(num_samples, num_plots, d, 1)`.
    """
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["global"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    keys = jax.random.split(rng_key, int(num_samples))
    generate_one = jax.jit(lambda k: sampler.generate(rng=k, num_samples=num_plots))

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all


def sample_wan_guided(
    diffusion_scheme,
    denoise_fn,
    y_bar: jnp.ndarray,
    rng_key: jax.Array,
    num_samples: int,
    run_sett,
):
    """ """

    C_prime = _build_C_prime(
        int(run_sett["global"]["d"]), run_sett["global"]["d_prime"]
    )

    guidance_transform = LinearConstraint.create(
        C_prime=C_prime,
        y_bar=y_bar,
        norm_guide_strength=run_sett["train_denoiser"]["norm_guide_strength"],
    )
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["global"]["d"], 1),
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

    keys = jax.random.split(rng_key, num_samples)
    generate_one = jax.jit(
        lambda k: sampler.generate(rng=k, num_samples=int(y_bar.shape[0]))
    )

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all


def sample_pde_guided(
    diffusion_scheme,
    denoise_fn,
    pde_solver,
    rng_key: jax.Array,
    samples_per_condition: int,
    y: jnp.ndarray,
):
    """ """
    num_conditionings = int(pde_solver.num_conditionings)
    if y.shape[0] != num_conditionings:
        raise ValueError(
            f"`y` must have leading size {num_conditionings}, but got {y.shape[0]}"
        )
    sampler = NewDriftSdeSampler(
        input_shape=(pde_solver.run_sett_global["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(pde_solver.run_sett_exp_tspan["num_steps"]),
            end_sigma=float(pde_solver.run_sett_exp_tspan["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        guidance_fn=pde_solver.grad_log_h_batched,
        apply_denoise_at_end=True,
        return_full_paths=False,
    )

    keys = jax.random.split(rng_key, samples_per_condition)
    generate_one = jax.jit(
        lambda k: sampler.generate(
            rng=k, num_samples=num_conditionings, guidance_inputs={"y": y}
        )
    )

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all
