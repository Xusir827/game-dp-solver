# Game-DP Solver: Zero-Sum Cloud Resource Competition

This project implements a numerical solver for **two-player zero-sum discounted stochastic games** motivated by **cloud compute resource competition**.

The solver combines:
- **Dynamic Programming** (Shapley-style backups / value-iteration family), and
- **Optimization** (per-state **LP** minimax solves for zero-sum matrix games).

In addition to plain Value Iteration (VI), the project includes:
- **Modified Policy Iteration (MPI)**, and
- **Policy Iteration with exact policy evaluation (PI-exact)** via a linear system solve.

---

## Quick start

Run the three main numerical examples (end-to-end). Each script prints a short summary and saves figures to `figures/<experiment_name>/`.

```bash
python examples/ex1_deterministic.py
python examples/ex2_stochastic.py
python examples/ex3_asymmetry.py
```
## 1）Setup
Requirements
- `Python 3.9+`
- Recommended: macOS / Linux
## 2) Project structure

- `model.py`  
  Game model: states/actions/transition probabilities/reward.

- `matrix_game_lp.py`  
  Per-state LP solver for zero-sum matrix games: `solve_zero_sum_matrix_game(M)`.

- `shapley_vi.py`  
  DP solvers:
  - `shapley_value_iteration` (VI)
  - `modified_policy_iteration` (MPI)
  - `policy_iteration_exact` (PI-exact with linear-system evaluation)

- `simulate.py`  
  Rollout simulation using stationary policies.

- `viz.py`  
  Plot utilities and figure saving helpers.

- `examples/`  
  Experiment scripts (end-to-end) that generate figures under `figures/`.

---

## 3) Outputs

Figures are saved under:
- `figures/ex1_deterministic/`
- `figures/ex2_stochastic/`
- `figures/ex3_asymmetry/`

Additional analysis outputs:
- `figures/runtime_scaling/`
- `figures/runtime_scaling_vi_mpi_pi/`
- `figures/gamma_sweep/`
- `figures/uncertainty_sweep/`
- `figures/vi_vs_mpi/`

---

## 4) Algorithm comparisons and scaling experiments

### VI vs MPI vs PI-exact (single instance)
```bash
python examples/compare_vi_mpi_pi.py
```

### VI vs MPI across all three examples
```bash
python python examples/run_all_vi_vs_mpi.py
```

### Runtime scaling with state size (single method)
```bash
python python examples/runtime_scaling.py
```

### Runtime scaling: VI vs MPI vs PI-exact
```bash
python python examples/runtime_scaling_vi_mpi_pi.py
```

## 5) Sensitivity analysis
Gamma sweep (planning horizon)
```bash
python examples/gamma_sweep.py
```

Uncertainty strength sweep (fixed mean replenishment, increasing variance)
```bash
python examples/uncertainty_sweep.py
```