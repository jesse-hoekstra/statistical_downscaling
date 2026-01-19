"""Helper script to count parameters of a UNet configuration.

Run this to quickly sanity-check the parameter count for a given UNet setup.
Optionally verify a forward pass shape using `--verify_forward`.
"""

import os
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.denoiser_utils import create_denoiser_model


def add_repo_root_to_path() -> None:
    """Ensure repository root is at the front of `sys.path`."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def human_count(n: int) -> str:
    """Format an integer count with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def count_params(pytree) -> int:
    """Count total number of scalar parameters in a JAX pytree."""
    sizes = [int(np.prod(np.array(x.shape))) for x in jax.tree_util.tree_leaves(pytree)]
    return int(sum(sizes))


def build_model_and_params():
    """Construct a UNet and initialize parameters for a fixed input shape.

    Returns a tuple of (model, variables) where `variables` contains a 'params'
    collection suitable for counting.
    """
    denoiser_model = create_denoiser_model()

    key = jax.random.PRNGKey(0)
    x = jnp.zeros((1, 192, 1))
    sigma = jnp.ones((1,))
    variables: dict[str, Any] = denoiser_model.init(
        {"params": key}, x, sigma, cond=None, is_training=False
    )
    return denoiser_model, variables


def main():
    """Parse args, build UNet, and report parameter count (and optional shape)."""
    add_repo_root_to_path()

    model, variables = build_model_and_params()

    params = variables.get("params", {})
    total = count_params(params)

    print("Model:", model.__class__.__name__)
    print()
    print("Config:")
    print(f"Total parameters: {total} ({human_count(total)})")


if __name__ == "__main__":
    main()
