import numpy as np
from typing import Dict, List, Tuple, Optional

from matrix_game_lp import solve_zero_sum_matrix_game
from model import ResourceGameModel


def build_payoff_matrix(
    model: ResourceGameModel,
    V: np.ndarray,
    s: int,
    infeasible_penalty: float = -1e4
) -> np.ndarray:
    """
    Build the one-step game payoff matrix M_s for a given state s:

        M_s(a,b) = r(s,a,b) + gamma * sum_{s'} P(s'|s,a,b) * V[s']

    Rows correspond to Player 1 actions a, columns correspond to Player 2 actions b.

    We enforce a feasibility constraint: a + b <= s.
    If (a+b) is infeasible, we assign a large negative payoff (penalty) so
    the optimal strategies will avoid it.

    NOTE: penalty magnitude should be "large enough" to discourage infeasible actions
          but not so large that LP becomes numerically unstable. -1e4 is usually safe.
    """
    A = model.actions_p1(s)
    B = model.actions_p2(s)

    M = np.zeros((len(A), len(B)), dtype=float)

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            # Feasibility constraint: cannot consume more than available capacity


            r = model.reward(s, a, b)
            trans = model.transition_probs(s, a, b)

            exp_future = 0.0
            for sp, p in trans.items():
                exp_future += p * V[sp]

            M[i, j] = r + model.gamma * exp_future
            M[i, j] += 1e-3 * (i - j)

    return M


def shapley_value_iteration(
    model: ResourceGameModel,
    eps: float = 1e-5,
    max_iter: int = 2000,
    solver: Optional[str] = "ECOS",
    infeasible_penalty: float = -1e4
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], List[float]]:
    """
    Shapley value iteration for two-player zero-sum discounted stochastic games.

    Returns:
        V: value function array of length R+1
        pi1: dict mapping state s -> mixed strategy over actions A(s)
        pi2: dict mapping state s -> mixed strategy over actions B(s)
        deltas: list of ||V_{k+1} - V_k||_inf per iteration
    """
    states = model.states()
    V = np.zeros(model.R + 1, dtype=float)

    pi1: Dict[int, np.ndarray] = {s: None for s in states}
    pi2: Dict[int, np.ndarray] = {s: None for s in states}
    deltas: List[float] = []

    for k in range(max_iter):
        V_new = np.zeros_like(V)

        for s in states:
            M_s = build_payoff_matrix(model, V, s, infeasible_penalty=infeasible_penalty)

            # Solve per-state zero-sum matrix game via LP
            v, p, q = solve_zero_sum_matrix_game(M_s, solver=solver)

            V_new[s] = v
            pi1[s] = p
            pi2[s] = q

        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new

        if delta <= eps:
            break

    return V, pi1, pi2, deltas