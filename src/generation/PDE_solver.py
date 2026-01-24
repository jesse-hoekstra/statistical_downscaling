"""Base PDE solver utilities built on JAX.

Defines `PDE_solver`, a base class providing network setup, sampling utilities,
training loop scaffolding, and gradient helpers. Concrete solvers should
subclass it and implement `loss_fn`.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial
from src.generation.DGMJax import DGMNetJax
import orbax.checkpoint as ocp
import os


def _build_C_prime(d: int, d_prime: int) -> jax.Array:
    downsampling_factor = d // d_prime

    return jnp.array(
        [
            [1 if j == downsampling_factor * i else 0 for j in range(d)]
            for i in range(d_prime)
        ]
    ).astype(jnp.float32)


class PDE_solver:
    """Base class for PDE solvers.

    Provides common initialization, sampling, forward pass, training loop,
    and gradient utilities. Subclasses must implement `loss_fn`.
    """

    def __init__(self, settings: dict, rng_key: jax.Array | None = None):
        """Initialize solver configuration, model parameters, and optimizer.

        Args:
            settings: Dict with sections `general`, `pde_solver`, and `DGM`
                specifying domain bounds, sampling sizes, optimizer
                hyperparameters, and network architecture.
            rng_key: Optional PRNG key for deterministic init and sampling.
        """
        # Common settings
        self.run_sett_pde_solver = settings["pde_solver"]
        self.run_sett_global = settings["global"]
        self.run_sett_dgm = settings["DGM"]
        self.run_sett_exp_tspan = settings["exp_tspan"]
        self.run_sett_ema = settings["ema"]

        self.seed = int(self.run_sett_global["seed"])
        self.T = float(self.run_sett_global["T"])
        self.d = int(self.run_sett_global["d"])
        self.d_prime = int(self.run_sett_global["d_prime"])
        self.time_emb_dim = int(self.run_sett_global["time_emb_dim"])

        self.C_prime = _build_C_prime(self.d, self.d_prime)

        self.ema_decay = float(self.run_sett_ema["ema_decay"])
        self.use_ema_eval = bool(self.run_sett_ema["use_ema_eval"])

        self.t_low = float(self.run_sett_pde_solver["t_low"])
        self.x_low = float(self.run_sett_pde_solver["x_low"])
        self.x_high = float(self.run_sett_pde_solver["x_high"])
        self.y_low = float(self.run_sett_pde_solver["y_low"])
        self.y_high = float(self.run_sett_pde_solver["y_high"])
        self.nSim_interior = int(self.run_sett_pde_solver["nSim_interior"])
        self.nSim_terminal = int(self.run_sett_pde_solver["nSim_terminal"])
        self.sampling_stages = int(self.run_sett_pde_solver["sampling_stages"])
        self.num_conditionings = int(self.run_sett_pde_solver["num_conditionings"])
        self.chunk_size = int(self.run_sett_pde_solver["chunk_size"])

        # DGM network
        hidden = int(self.run_sett_dgm["nodes_per_layer"])
        n_layers = int(self.run_sett_dgm["num_layers"])
        self.net = DGMNetJax(
            input_dim=self.d + self.d_prime + self.time_emb_dim,
            time_emb_dim=self.time_emb_dim,
            layer_width=hidden,
            num_layers=n_layers,
            C_prime=self.C_prime,
            final_trans=None,
        )

        # Initialize parameters with dummy batch
        t0 = jnp.zeros((1, 1), dtype=jnp.float32)
        x0 = jnp.zeros((1, self.d), dtype=jnp.float32)
        y0 = jnp.zeros((1, self.d_prime), dtype=jnp.float32)
        self.params = self.net.init(jax.random.PRNGKey(self.seed), t0, x0, y0)

        self.ema_params = jax.tree_util.tree_map(lambda x: x, self.params)

        # Optimizer
        self.learning_rate = self._build_lr_schedule()
        self.optimizer = optax.adam(self.learning_rate)

        self.opt_state = self.optimizer.init(self.params)

        self._step = 0

    def _build_lr_schedule(self):
        """Create learning-rate schedule (constant or cosine with warmup/decay)."""
        boundaries = self.run_sett_pde_solver["boundaries"]
        constant_lr = float(self.run_sett_pde_solver["constant_lr"])
        lr_schedules = self.run_sett_pde_solver["lr_schedules"]
        mode_type = str(self.run_sett_pde_solver["type_lr_schedule"]).lower()
        if mode_type == "constant":
            return optax.constant_schedule(constant_lr)
        elif mode_type == "sirignano":
            schedules = [optax.constant_schedule(float(lr)) for lr in lr_schedules]
            return optax.join_schedules(schedules=schedules, boundaries=boundaries)

    def sampler(self, key):
        """Sample interior and terminal training points for the PDE.

        Returns:
            Tuple `(t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal)` where
            interior samples are uniform in time over `(t_low, T]` and uniform
            in space over `[x_low, x_high]`, and terminal times are fixed at `T`.
            Interior targets are sampled from a uniform distribution over `[x_low, x_high]`.
            Terminal targets are sampled from a uniform distribution over `[x_low, x_high]`.
        """
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        split_ratio = 0.5  # to ensure it sees a part not close to manifold as well

        num_coupled = int(self.nSim_interior * split_ratio)
        num_uniform = self.nSim_interior - num_coupled

        t_interior = jax.random.uniform(
            k1, (self.nSim_interior, 1), minval=self.t_low, maxval=self.T
        )
        x_interior = jax.random.uniform(
            k2, (self.nSim_interior, self.d), minval=self.x_low, maxval=self.x_high
        )

        std_scale_interior = 0.005 * jnp.std(x_interior)
        y_int_coupled = (
            x_interior[:num_coupled] @ self.C_prime.T
        ) + std_scale_interior * jax.random.normal(k3, (num_coupled, self.d_prime))
        y_int_uniform = jax.random.uniform(
            k4, (num_uniform, self.d_prime), minval=self.y_low, maxval=self.y_high
        )
        y_interior = jnp.concatenate([y_int_coupled, y_int_uniform], axis=0)

        num_term_coupled = int(self.nSim_terminal * split_ratio)
        num_term_uniform = self.nSim_terminal - num_term_coupled

        t_terminal = jnp.ones((self.nSim_terminal, 1)) * self.T
        x_terminal = jax.random.uniform(
            k5, (self.nSim_terminal, self.d), minval=self.x_low, maxval=self.x_high
        )

        std_scale_terminal = 0.005 * jnp.std(x_terminal)
        y_term_coupled = (
            x_terminal[:num_term_coupled] @ self.C_prime.T
        ) + std_scale_terminal * jax.random.normal(k6, (num_term_coupled, self.d_prime))
        y_term_uniform = jax.random.uniform(
            k7, (num_term_uniform, self.d_prime), minval=self.y_low, maxval=self.y_high
        )
        y_terminal = jnp.concatenate([y_term_coupled, y_term_uniform], axis=0)

        return t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal

    @partial(jax.jit, static_argnums=0)
    def V(self, params, t: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        """Evaluate the value network `V(t, x, y)`.

        Args:
            params: Network parameters.
            t: Time tensor of shape `(B, 1)`.
            x: Spatial tensor of shape `(B, d)`.
            y: Target tensor of shape `(B, d_prime)`.

        Returns:
            Tensor of shape `(B, 1)` with the network output.
        """
        return self.net.apply(params, t, x, y)

    @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, t_interior, x_interior, t_terminal, x_terminal):
        """Compute loss terms. Must be implemented by subclasses.

        Returns:
            A tuple `(L1, L3, total)` of scalar losses.
        """
        raise NotImplementedError("loss_fn must be implemented by subclasses")

    def _train_step(self, key):
        """Train the network using SGD/Adam over sampled batches.

        Iterates for `sampling_stages`, drawing fresh samples each stage and
        performing `steps_per_sample` optimization steps per model.

        Args:
            log_fn: Optional callable receiving a dict of scalar metrics once
                per sampling stage.
        """
        batch = self.sampler(key)
        chunk_size = int(self.chunk_size)
        num_chunks = self.nSim_interior // chunk_size

        def loss_and_aux(p, *batch_slice):
            return self.loss_fn(p, *batch_slice)

        accumulated_grads = None
        total_aux = None

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            batch_slice = (arr[start:end] for arr in batch)
            (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
                self.params, *batch_slice
            )
            if accumulated_grads is None:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda g: g / num_chunks, grads
                )
                total_aux = {k: v / num_chunks for k, v in aux.items()}
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda acc, g: acc + (g / num_chunks), accumulated_grads, grads
                )
                for k in total_aux:
                    total_aux[k] += aux[k] / num_chunks

        updates, self.opt_state = self.optimizer.update(
            accumulated_grads, self.opt_state
        )
        self.params = optax.apply_updates(self.params, updates)

        self.ema_params = jax.tree_util.tree_map(
            lambda e, p: self.ema_decay * e + (1.0 - self.ema_decay) * p,
            self.ema_params,
            self.params,
        )

        return total_aux

    def update_params(self, key):
        """Advance training by one step and return scalar metrics."""
        metrics = self._train_step(key)
        self._step += 1
        return {k: float(v) for k, v in metrics.items()}

    @partial(jax.jit, static_argnums=0)
    def grad_log_h_params(
        self, params: jax.Array, x: jax.Array, y: jax.Array, t: jax.Array
    ) -> jax.Array:
        """Compute per-sample gradients ∂/∂x log h(t, x) for given params.

        Returns an array matching the flattened spatial dimension of `x`.
        """
        x_flat = x.reshape(1, -1)
        y_flat = y.reshape(1, -1)
        t = t.reshape(1, 1)

        def loss(xs, ys):
            return self.net.apply(params, t, xs, ys).squeeze()  # outputs log directly.

        return jax.grad(loss)(x_flat, y_flat)

    @partial(jax.jit, static_argnums=0)
    def grad_log_h(self, x: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Compute `grad_log_h`."""
        return self.grad_log_h_params(self.ema_params, x, y, t)

    @partial(jax.jit, static_argnums=0)
    def grad_log_h_batched(self, x: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Compute grad_log_h for a batch where sample i uses conditioning i.

        Args:
            x: Array `(M, d)` or `(M, d, 1)` where `M == self.num_conditionings`.
            y: Array `(M, d_prime)` or `(M, d_prime, 1)` where `M == self.num_conditionings`.
            t: Scalar `()` or array `(M,)` or `(M, 1)`.

        Returns:
            Array `(M, d)` containing per-sample gradients.
        """
        M = x.shape[0]
        x_flat = x.reshape(M, -1)
        y_flat = y.reshape(M, -1)
        if t.ndim == 0:
            t = jnp.ones((M, 1), dtype=x_flat.dtype) * t
        elif t.ndim == 1:
            t = t.reshape(M, 1)

        def per_sample(xs, ys, ts):
            return self.grad_log_h_params(self.ema_params, xs, ys, ts)

        grads = jax.vmap(per_sample)(x_flat, y_flat, t)
        if grads.ndim == 3 and grads.shape[1] == 1:
            grads = jnp.squeeze(grads, axis=1)
        return grads

    def compute_ema_metrics(self, key):
        """Compute metrics using EMA params."""
        batch = self.sampler(key)

        chunk_size = int(self.chunk_size)
        num_chunks = self.nSim_interior // chunk_size
        total_ema_aux = None

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            batch_slice = tuple(arr[start:end] for arr in batch)
            _, aux = self.loss_fn(self.ema_params, *batch_slice)

            if total_ema_aux is None:
                total_ema_aux = {k: v / num_chunks for k, v in aux.items()}
            else:
                for k in total_ema_aux:
                    total_ema_aux[k] += aux[k] / num_chunks

        return {f"eval/{k}": float(v) for k, v in total_ema_aux.items()}

    def save_params(self, ckpt_dir: str):
        """Save current params_list and opt_state_list to a checkpoint directory.

        Args:
            ckpt_dir: Directory path to write the checkpoint. This directory will be
                created if it doesn't exist, and its contents will be overwritten.
        """
        checkpointer = ocp.StandardCheckpointer()
        options = ocp.CheckpointManagerOptions(create=True, max_to_keep=2)
        mngr = ocp.CheckpointManager(
            os.path.abspath(ckpt_dir), checkpointer, options=options
        )
        payload = {
            "params": self.params,
            "ema_params": self.ema_params,
            "opt_state": self.opt_state,
        }
        current_step = self._step
        mngr.save(step=current_step, args=ocp.args.StandardSave(payload))
        final_path = os.path.join(ckpt_dir, str(current_step))
        print(f"Solver state saved successfully to: {final_path}")

    def load_params(self, ckpt_dir: str) -> bool:
        """Load the latest params_list and opt_state_list using Orbax CheckpointManager."""
        try:
            mngr = ocp.CheckpointManager(
                os.path.abspath(ckpt_dir), ocp.StandardCheckpointer()
            )
            latest_step = mngr.latest_step()
            if latest_step is None:
                print(f"No checkpoints found in: {ckpt_dir}")
                return False
            target_item = {
                "params": self.params,
                "ema_params": self.ema_params,
                "opt_state": self.opt_state,
            }

            restored = mngr.restore(
                latest_step, args=ocp.args.StandardRestore(target_item)
            )

            self.params = restored["params"]
            self.ema_params = restored["ema_params"]
            self.opt_state = restored["opt_state"]

            self._step = latest_step

            print(
                f"Solver state loaded successfully from step {latest_step} at: {ckpt_dir}"
            )

        except Exception as e:
            print(f"Failed to load solver state. Error: {e}")
