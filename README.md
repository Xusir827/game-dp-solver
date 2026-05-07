# Game-DP Solver: Zero-Sum Cloud Resource Competition

This project implements a numerical solver for **two-player zero-sum discounted stochastic games** motivated by **cloud compute resource competition**.  
The solver combines:
- **Dynamic Programming** (Shapley-style backups / value iteration family), and
- **Optimization** (per-state **LP** minimax solves for zero-sum matrix games).

In addition to plain Value Iteration (VI), the project includes:
- **Modified Policy Iteration (MPI)**, and
- **Policy Iteration with exact policy evaluation (PI-exact)** via a linear system solve.

---

## 1) Setup

### Requirements
- Python 3.9+
- Recommended: macOS / Linux

### Install dependencies
Create and activate a virtual environment, then install packages:

```bash
pip install numpy matplotlib cvxpy ecos osqp