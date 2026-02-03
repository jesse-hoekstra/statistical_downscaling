"""Deep Galerkin Method (DGM) components implemented in JAX.

Provides a recurrent DGM block (`LSTMLayerJax`), an affine layer with optional
nonlinearity (`DenseLayerJax`), and a full network (`DGMNetJax`) that stacks
these components to approximate PDE solutions.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class LSTMLayerJax(nn.Module):
    """Recurrent block used in Deep Galerkin Method (DGM).

    Implements gated updates resembling LSTM to evolve a hidden state given
    the current concatenated input [t, x].
    """

    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, S: jax.Array, X: jax.Array) -> jax.Array:
        """One recurrent block used in Deep Galerkin Networks.

        Args:
            S: Hidden state tensor of shape `(B, output_dim)`.
            X: Input tensor of shape `(B, input_dim)`; here usually `[t, x]`.

        Returns:
            Updated hidden state tensor `(B, output_dim)`.
        """
        glorot = nn.initializers.glorot_uniform()
        Uz = self.param("Uz", glorot, (self.input_dim, self.output_dim))
        Ug = self.param("Ug", glorot, (self.input_dim, self.output_dim))
        Ur = self.param("Ur", glorot, (self.input_dim, self.output_dim))
        Uh = self.param("Uh", glorot, (self.input_dim, self.output_dim))

        Wz = self.param("Wz", glorot, (self.output_dim, self.output_dim))
        Wg = self.param("Wg", glorot, (self.output_dim, self.output_dim))
        Wr = self.param("Wr", glorot, (self.output_dim, self.output_dim))
        Wh = self.param("Wh", glorot, (self.output_dim, self.output_dim))

        bz = self.param("bz", nn.initializers.zeros, (self.output_dim,))
        bg = self.param("bg", nn.initializers.zeros, (self.output_dim,))
        br = self.param("br", nn.initializers.zeros, (self.output_dim,))
        bh = self.param("bh", nn.initializers.zeros, (self.output_dim,))

        Z = jnp.tanh(jnp.matmul(X, Uz) + jnp.matmul(S, Wz) + bz)
        G = jnp.tanh(jnp.matmul(X, Ug) + jnp.matmul(S, Wg) + bg)
        R = jnp.tanh(jnp.matmul(X, Ur) + jnp.matmul(S, Wr) + br)
        H = jnp.tanh(jnp.matmul(X, Uh) + jnp.matmul(S * R, Wh) + bh)
        S_new = (1.0 - G) * H + Z * S
        return S_new


class DenseLayerJax(nn.Module):
    """Affine layer with optional nonlinearity."""

    input_dim: int
    output_dim: int
    transformation: Optional[str] = None  # 'tanh', 'relu', or None

    @nn.compact
    def __call__(self, X: jax.Array) -> jax.Array:
        """Affine + optional nonlinearity layer.

        Args:
            X: Input tensor `(B, input_dim)`.

        Returns:
            Tensor `(B, output_dim)` after linear layer and optional activation.
        """
        W = self.param(
            "W", nn.initializers.glorot_uniform(), (self.input_dim, self.output_dim)
        )
        b = self.param("b", nn.initializers.zeros, (self.output_dim,))
        S = jnp.matmul(X, W) + b
        if self.transformation == "tanh":
            S = jnp.tanh(S)
        elif self.transformation == "relu":
            S = nn.relu(S)
        return S


class DGMNetJax(nn.Module):
    """Deep Galerkin Method (DGM) implemented with Flax.

    Stacks an initial dense layer, multiple recurrent DGM blocks, and a final
    projection to produce a scalar PDE solution estimate.
    """

    input_dim: int  # spatial input dimension d + d_prime + time_emb_dim
    time_emb_dim: int
    layer_width: int
    num_layers: int
    C_prime: jax.Array
    final_trans: Optional[str] = None

    def _sinusoidal_time_embedding(self, n: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Return sinusoidal time embedding for step index `n`.

        This mirrors positional encodings: for frequencies geometrically spaced
        between [1, 1/10000] we apply sin/cos and concatenate.

        Parameters
        ----------
        n : jnp.ndarray
            Scalar or array of time indices.
        dim : int
            Embedding dimension.

        Returns
        -------
        jnp.ndarray
            Embedding with shape `n.shape + (dim,)`.
        """
        dim = int(dim)
        half = dim // 2
        _fdtype = jnp.asarray(0.0).dtype
        n = n.astype(_fdtype)
        freqs = jnp.exp(
            -jnp.log(jnp.asarray(10000.0, dtype=_fdtype))
            * jnp.arange(0, half, dtype=_fdtype)
            / jnp.maximum(half, 1)
        )
        args = n[..., None] * freqs[None, ...]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if dim % 2 == 1:
            emb = jnp.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
        return emb

    @nn.compact
    def __call__(self, t: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        """Forward pass of a Deep Galerkin Network (DGM) in JAX/Flax. With added sinusoidal time embedding and gated connections.

        Args:
            t: Time inputs `(B, 1)`.
            x: Spatial inputs `(B, d)`.
            y: Target inputs `(B, d_prime)`.
        Returns:
            Scalar output `(B, 1)` implementing an approximation to the PDE solution.
        """
        te = self._sinusoidal_time_embedding(t, self.time_emb_dim).reshape(
            t.shape[0], -1
        )
        y_feat = jnp.concatenate([te, y], axis=1)  # no residuals
        S_y = DenseLayerJax(y_feat.shape[1], self.layer_width, transformation="tanh")(
            y_feat
        )
        y_gate = jax.nn.sigmoid(S_y)

        x_feat = jnp.concatenate([te, x], axis=1)
        S_x = DenseLayerJax(x_feat.shape[1], self.layer_width, transformation="tanh")(
            x_feat
        )
        X = y_gate * S_x + (1 - y_gate) * S_y

        S = DenseLayerJax(self.layer_width, self.layer_width, transformation="tanh")(X)
        for _ in range(self.num_layers):
            S = LSTMLayerJax(self.layer_width, self.layer_width)(S, X)
        out = DenseLayerJax(self.layer_width, 1, transformation=self.final_trans)(S)
        return out
