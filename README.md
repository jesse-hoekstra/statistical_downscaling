# Statistical Downscaling Project

A Python implementation of statistical downscaling. This project is divided in conditional generation and optimal transport. 

## Installation

### Prerequisites
- Python 3.12.8 

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Statistical_Downscaling
   ```

### Optimal Transport

2. **Set working directory**
   ```bash
   python os.chdir(os.path.join(os.getcwd(), "src", "optimal_transport"))
   ```

3. **Create virtual environment**
   ```bash
   python -m venv .venv_OT
   ```

4. **Activate the environment**
   ```bash
   # On macOS/Linux:
   source .venv_OT/bin/activate
   
   # On Windows:
   .venv_OT\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements_OT.txt
   ```

### Generation

2. **Set working directory**
   ```bash
   python os.chdir(os.path.join(os.getcwd(), "src", "generation"))
   ```

3. **Create virtual environment**
   ```bash
   python -m venv .venv_GEN
   ```

4. **Activate the environment**
   ```bash
   # On macOS/Linux:
   source .venv_GEN/bin/activate
   
   # On Windows:
   .venv_GEN\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements_GEN.txt
   ```

## Project Structure

```
Statistical_Downscaling/
├── README.md               # This file
├── src/                    # Source code modules
│   ├── generation/
│   └── optimal_transport/            
├── output/                 # Output results directory
└── scripts/                # Utility scripts
    └── pre_commit.sh       # Code formatting script
```

## Configuration

### Optimal Transport

The project uses `settings_OT.yaml` for configuration:

```yaml
global:
  seed: 0
  N: 3
  d: 2
  B: 128
  time_emb_dim: 16
  num_iterations: 30000
  RNG_NAMESPACE: 2026000000
marginal_flow:
  num_bins: 8
  num_layers: 4
  hidden_size: 128
  # need to calculate these specifically data-based!!
  range_min: -4.0
  range_max: 4.0
correlation_flow:
  rho_max: 0.95
  rho_hidden: 64
  hidden_size: 128
beta_schedule:
  type: "constant"
  warmup_end_ratio: 0.2
  init_boundary_value: 0.0
  end_boundary_value: 2.0
  kl_only_warmup_steps: 0
lr_schedule:
  type: "constant"
  constant_lr: 1e-3
  init_value: 0.0
  peak_value: 0.0003
  warmup_steps: 10000
  decay_steps: 40000
  end_value: 0.000001
ema: 
  ema_decay: 0.999
  use_ema_eval: True
preprocessing:
  use_data_normalization: True
  winsor_clip_z: 10.0
  num_samples: 4096
  chunk_size: 512
  mode: "time_varying"
  eps: 1e-6
  RNG_NAMESPACE_NORM: 20260113
baseline_fitting:
  cv_ridge: 1e-3
  cv_split_ratio: 0.0
policy_gradient:
  mix_pathwise_alpha: 0.0
  cv_split_ratio: 0.5
  use_control_variates: True
  use_advantage_standardization: True
  RNG_NAMESPACE_PG: 2026
metrics:
  chunk_size: 256
  num_samples: 2048
  log_scalar_every: 200
  log_adjcorr_every: 3000
  adjcorr_first_k: 3
  log_hist_every: 3000
  log_hist_n: 4
  log_train_every: 50
  plot_samples_max: 2048
  adjcorr_max_samples: 2048
wandb:
  use_wandb: True
```

### Generation

The project uses `settings_GEN.yaml` for configuration:

```yaml
global:
  seed: 888
  RNG_NAMESPACE: 20260115
  mode: 'sample'                    # 'train' | 'sample' | 'eval'
  generation_type: 'conditional'    # 'unconditional' | 'wan_conditional' | 'conditional'
  train_denoiser: False
  train_pde: True
  data_file_name: 'data/ks_trajectories_512.h5'
  d: 192
  d_prime: 24
  T: 1
  output_dir: 'src/generation/output/'
  chunk_size_sampler: 16
  hyperparameter_tuning: True
train_denoiser:
  batch_size: 512
  total_train_steps: 1_000_000
  eval_every_steps: 100
  metric_aggregation_steps: 100
  save_interval_steps: 100
  num_batches_per_eval: 10
  max_to_keep: 1
  beta_min: 0.1
  beta_max: 20
  norm_guide_strength: 1.0
  ema_decay: 0.95
  clip_min: 1e-3
UNET:
  out_channels: 1
  num_channels: 32,64,128
  downsample_ratio: 2,2,2
  num_blocks: 6
  noise_embed_dim: 128
  use_attention: True
  num_heads: 8
  use_position_encoding: False
  dropout_rate: 0.0
optimizer:
  init_value: 0.0
  peak_value: 1e-3
  warmup_steps: 1_000
  decay_steps: 990_000
  end_value: 1e-6
  clip_norm: 1.0
  clip_gradient: 200_000.0
  use_clip_gradient: False
exp_tspan:
  num_steps: 256
  end_sigma: 1e-2
pde_solver:
  t_low: 1e-10
  nSim_interior: 1000
  nSim_terminal: 1000
  x_low: -5.0
  x_high: 5.0
  y_low: -5.0
  y_high: 5.0
  sampling_stages: 75_000
  type_lr_schedule: sirignano
  constant_lr: 1e-3
  boundaries: [25_000, 50_000]
  lr_schedules: [1e-3, 5e-4, 1e-4]
  lambda: 1
  num_gen_samples: 128
  num_conditionings: 512
  chunk_size: 250
  adaptive_balancing_loss: False
  lambda_smooth_alpha: 0.1
  normalize_data: False
DGM:
  nodes_per_layer: 256
  num_layers: 5
ema:
  ema_decay: 0.999
  use_ema_eval: True
wandb:
  use_wandb: False
metrics:
  log_ema_metrics_every: 500
  log_train_metrics_every: 50
  epsilon: 1e-5
```

## Usage

### Basic Execution

#### Optimal Transport

Run the main script:

```bash
python main_OT.py
```

#### Generation

Run the main script:

```bash
python main_GEN.py
```

Common workflows:
- Train denoiser only:
  1) In `src/generation/settings_GEN.yaml`: set `global.mode: 'train'`, `global.train_denoiser: True`, `global.train_pde: False`.
  2) Run:
     ```bash
     python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
     ```
- Train PDE solver (requires a denoiser checkpoint in the same `work_dir`):
  1) Set `global.mode: 'train'`, `global.train_denoiser: False`, `global.train_pde: True`.
     - Alternatively, set both True to first train the denoiser then the PDE solver in one run.
  2) Run the same command as above.
- Sample/generate:
  1) Set `global.mode: 'sample'` and choose `global.generation_type`:
     - `unconditional`: draws from the learned prior.
     - `wan_conditional`: LinearConstraint guidance using the denoiser; optionally enable `global.hyperparameter_tuning` to sweep `norm_guide_strength` and `exp_tspan.num_steps`.
     - `conditional`: PDE-guided sampling using the learned guidance function; requires PDE checkpoints.
  2) Run:
     ```bash
     python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
     ```
  3) Samples are saved to `main_GEN/<run_name>/samples_<generation_type>.h5`.
- Evaluate metrics on saved samples:
  1) Set `global.mode: 'eval'` and keep `global.generation_type` consistent with the saved sample file.
  2) Run:
     ```bash
     python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
     ```
  3) Prints and (optionally) logs: constraint RMSE, KLD, sample variability, MELR (weighted/unweighted), and 1-Wasserstein.

### Custom Configuration

Modify the `settings_X.yaml` files to experiment with different parameters:

## Key Components

### Optimal Transport 

#### `src/optimal_transport/main_OT.py`
- Orchestrates experiments from `--config` (default `src/optimal_transport/settings_OT.yaml`), sets a per-run `work_dir` under `main_OT/`, and manages logging.
- Instantiates `TrueDataModelUnimodal`, `NormalizingFlowModel`, and `PolicyGradient`, then runs the training loop for `num_iterations`.
- Integrates W&B via `src/generation/wandb_adapter.WandbWriter` when `wandb.use_wandb=True`.
- Logs scalar metrics, adjacent correlations, and histogram comparisons using helpers from `utils_OT.py`.

#### `src/optimal_transport/alg1_OT.py`
- Core components:
  - `sinusoidal_time_embedding` for time conditioning.
  - `ConditionalSplineCouplingFlow` implementing RQS masked coupling flows.
  - `RhoNet` to produce time- and state-dependent Gaussian-copula correlations.
  - `NormalizingFlowModel` that samples sequences and computes log-likelihoods in normalized space, with optional data normalization.
  - `PolicyGradient` trainer mixing pathwise transport cost and a score-function surrogate, with EMA evaluation, baseline fitting, and configurable beta/lr schedules.

#### `src/optimal_transport/preprocessing_OT.py`
- `DataNormalizer` that stream-estimates per-time or global statistics, applies winsor-clipped normalization/denormalization, and supplies log-det terms for change-of-variables.

#### `src/optimal_transport/dgp_OT.py`
- Synthetic true-data generators:
  - `TrueDataModelUnimodal`
  - `TrueDataModelBimodal`
- Each produces paired trajectories `(y, y')` for optimal transport experiments.

#### `src/optimal_transport/utils_OT.py`
- Plotting and diagnostics utilities (e.g., adjacent correlations, trajectory comparisons) used by `main_OT.py` for periodic logging.

### Generation

- `src/generation/main_GEN.py`: Entry point. Reads `--config`, constructs a per-run `work_dir` under `main_GEN/`, sets up optional W&B logging, and executes one of:
  - `mode: 'train'` → trains denoiser (`train_denoiser=True`) and/or PDE solver (`train_pde=True`).
  - `mode: 'sample'` → runs `generation_type` in {'unconditional','wan_conditional','conditional'} and writes `samples_<generation_type>.h5`.
  - `mode: 'eval'` → computes metrics from `utils_metrics.py` on saved samples.
- `src/generation/denoiser_utils.py`: Builds the UNet denoiser, VP diffusion scheme, trainer, and restores an EMA `denoise_fn` from Orbax checkpoints.
- `src/generation/sampler_utils.py`: Sampling helpers:
  - `sample_unconditional`,
  - `sample_wan_guided` (LinearConstraint guidance),
  - `sample_pde_guided` (PDE-driven guidance via `NewDriftSdeSampler`).
- `src/generation/Statistical_Downscaling_PDE_KS.py`: KS-specific solver that learns a value function to provide gradients of `log h(t,x,y)` for conditional guidance.
- `src/generation/PDE_solver.py`: Base solver (DGM network, training loop, EMA, checkpointing, gradient utilities).
- `src/generation/utils_metrics.py`: Metrics (constraint RMSE, KLD, MELR weighted/unweighted, 1-Wasserstein, sample variability).
- `src/generation/data_utils.py`: Loads HF/LF KS datasets from HDF5 and builds deterministic training/eval pipelines.
- `wandb_integration/wandb_adapter.py`: `WandbWriter` adapter that mirrors `clu.metric_writers` to Weights & Biases. Enable via `wandb.use_wandb: True`. You can also set:
  - `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME` (optional),
  - `WANDB_DISABLED=1` to force-disable,
  - `GPU_TAG` to suffix the run name for device tagging.

## Development

### Code Formatting

The project includes automated code formatting:

1. **Setup pre-commit hook**:
   ```bash
   # Add to .git/hooks/pre-commit:
   #!/bin/sh
   source scripts/pre_commit.sh
   ```

2. **Make executable**:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

### Adding New Features

#### Optimal Transport

1. **Model Extensions**: Add new paramaterized and true models in `src/optimal_transport/alg1_OT.py` and `src/optimal_transport/dgp_OT.py`
2. **Utilities**: Add helper functions in `src/optimal_transport/preprocessing_OT.py`

