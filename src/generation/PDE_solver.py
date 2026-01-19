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

        self.seed = int(self.run_sett_global["seed"])
        self.T = float(self.run_sett_global["T"])
        self.d = int(self.run_sett_global["d"])
        self.d_prime = int(self.run_sett_global["d_prime"])

        self.t_low = float(self.run_sett_pde_solver["t_low"])
        self.x_low = float(self.run_sett_pde_solver["x_low"])
        self.x_high = float(self.run_sett_pde_solver["x_high"])
        self.y_low = float(self.run_sett_pde_solver["y_low"])
        self.y_high = float(self.run_sett_pde_solver["y_high"])
        self.nSim_interior = int(self.run_sett_pde_solver["nSim_interior"])
        self.nSim_terminal = int(self.run_sett_pde_solver["nSim_terminal"])
        self.sampling_stages = int(self.run_sett_pde_solver["sampling_stages"])
        self.steps_per_sample = int(self.run_sett_pde_solver["steps_per_sample"])
        self.num_conditionings = int(self.run_sett_pde_solver["num_conditionings"])

        # DGM network
        hidden = int(self.run_sett_dgm["nodes_per_layer"])
        n_layers = int(self.run_sett_dgm["num_layers"])
        self.net = DGMNetJax(
            input_dim=self.d + self.d_prime,
            layer_width=hidden,
            num_layers=n_layers,
            final_trans=None,
        )

        # Initialize parameters with dummy batch
        t0 = jnp.zeros((1, 1), dtype=jnp.float32)
        x0 = jnp.zeros((1, self.d), dtype=jnp.float32)
        y0 = jnp.zeros((1, self.d_prime), dtype=jnp.float32)
        self.params = self.net.init(jax.random.PRNGKey(self.seed), t0, x0, y0)
        self.opt_state = self.optimizer.init(self.params)

        # Optimizer
        self.learning_rate = self._build_lr_schedule()
        self.optimizer = optax.adam(self.learning_rate)

        self._step = 0

    def _build_lr_schedule(self):
        """Create learning-rate schedule (constant or cosine with warmup/decay)."""
        boundaries = self.settings["pde_solver"]["boundaries"]
        constant_lr = float(self.settings["pde_solver"]["constant_lr"])
        mode_type = str(self.settings["pde_solver"]["type_lr_schedule"]).lower()
        if mode_type == "constant":
            return optax.constant_schedule(constant_lr)
        elif mode_type == "sirignano":
            schedules = [
                optax.constant_schedule(1e-4),
                optax.constant_schedule(5e-4),
                optax.constant_schedule(1e-5),
                optax.constant_schedule(5e-6),
                optax.constant_schedule(1e-6),
                optax.constant_schedule(5e-7),
                optax.constant_schedule(1e-7),
            ]
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
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        t_interior = jax.random.uniform(
            k1, shape=(self.nSim_interior, 1), minval=self.t_low, maxval=self.T
        )
        x_interior = jax.random.uniform(
            k2,
            shape=(self.nSim_interior, self.d),
            minval=self.x_low,
            maxval=self.x_high,
        )
        y_interior = jax.random.uniform(
            k3,
            shape=(self.nSim_interior, self.d_prime),
            minval=self.x_low,
            maxval=self.x_high,
        )
        t_terminal = jnp.ones((self.nSim_terminal, 1), dtype=jnp.float32) * self.T
        x_terminal = jax.random.uniform(
            k4,
            shape=(self.nSim_terminal, self.d),
            minval=self.x_low,
            maxval=self.x_high,
        )
        y_terminal = jax.random.uniform(
            k5,
            shape=(self.nSim_terminal, self.d_prime),
            minval=self.y_low,
            maxval=self.y_high,
        )
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
        t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal = (
            self.sampler(key)
        )
        for _ in range(self.steps_per_sample):

            def loss_and_aux(p):
                return self.loss_fn(
                    p,
                    t_interior,
                    x_interior,
                    y_interior,
                    t_terminal,
                    x_terminal,
                    y_terminal,
                )

            (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
                self.params
            )

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)
        return aux

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
            h = self.net.apply(params, t, xs, ys).squeeze(-1)
            return jnp.sum(jnp.log(jnp.maximum(h, 1e-6)))

        return jax.grad(loss)(x_flat, y_flat)

    @partial(jax.jit, static_argnums=0)
    def grad_log_h(self, x: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Compute `grad_log_h`."""
        return self.grad_log_h_params(self.params, x, y, t)

    @partial(jax.jit, static_argnums=0)
    def grad_log_h_batched_one_per_model(
        self, x: jax.Array, y: jax.Array, t: jax.Array
    ) -> jax.Array:
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
            return self.grad_log_h_params(self.params, xs, ys, ts)

        grads = jax.vmap(per_sample)(x_flat, y_flat, t)
        if grads.ndim == 3 and grads.shape[1] == 1:
            grads = jnp.squeeze(grads, axis=1)
        return grads

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
                "opt_state": self.opt_state,
            }

            restored = mngr.restore(
                latest_step, args=ocp.args.StandardRestore(target_item)
            )

            self.params = restored["params"]
            self.opt_state = restored["opt_state"]

            self._step = latest_step

            print(
                f"Solver state loaded successfully from step {latest_step} at: {ckpt_dir}"
            )

        except Exception as e:
            print(f"Failed to load solver state. Error: {e}")
