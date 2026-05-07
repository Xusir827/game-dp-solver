import numpy as np
import cvxpy as cp
from typing import Optional, Tuple


def solve_zero_sum_matrix_game(M: np.ndarray, solver: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solve a zero-sum matrix game with payoff matrix M for the ROW player (Player 1).

    Game definition:
        - Player 1 chooses a row i (or mixed strategy p over rows)
        - Player 2 chooses a column j (or mixed strategy q over cols)
        - Player 1 receives payoff M[i, j]
        - Player 2 receives payoff -M[i, j]

    Returns:
        v: minimax value (expected payoff to Player 1 under optimal play)
        p: optimal mixed strategy for Player 1 (probabilities over rows)
        q: optimal mixed strategy for Player 2 (probabilities over cols)

    Notes:
        We solve two LPs:
          (1) Row player's maximin LP:
              maximize v
              s.t.  p^T M[:, j] >= v  for all columns j
                    sum(p) = 1,  p >= 0
          (2) Column player's minimax LP:
              minimize w
              s.t.  M[i, :] q <= w    for all rows i
                    sum(q) = 1,  q >= 0
    """
    M = np.asarray(M, dtype=float)
    m, n = M.shape  # m rows, n cols

    # ---------------------------
    # (1) Player 1 (row) LP
    # ---------------------------
    # p is a probability vector over rows, v is the guaranteed payoff
    p = cp.Variable(m, nonneg=True)
    v = cp.Variable()

    constraints = [cp.sum(p) == 1]
    # For every column j, expected payoff under p must be >= v
    constraints += [p @ M[:, j] >= v for j in range(n)]

    prob = cp.Problem(cp.Maximize(v), constraints)
    prob.solve(solver=solver)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Row-player LP failed: status={prob.status}")

    p_star = np.array(p.value).flatten()
    v_star = float(v.value)

    # ---------------------------
    # (2) Player 2 (column) LP
    # ---------------------------
    # q is a probability vector over columns, w is the upper bound on Player 1 payoff
    q = cp.Variable(n, nonneg=True)
    w = cp.Variable()

    constraints2 = [cp.sum(q) == 1]
    # For every row i, expected payoff under q must be <= w
    constraints2 += [M[i, :] @ q <= w for i in range(m)]

    prob2 = cp.Problem(cp.Minimize(w), constraints2)
    prob2.solve(solver=solver)

    if prob2.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Col-player LP failed: status={prob2.status}")

    q_star = np.array(q.value).flatten()

    # ---------------------------
    # Numerical cleanup
    # ---------------------------
    p_star = np.maximum(p_star, 0)
    q_star = np.maximum(q_star, 0)
    if p_star.sum() > 0:
        p_star = p_star / p_star.sum()
    if q_star.sum() > 0:
        q_star = q_star / q_star.sum()

    return v_star, p_star, q_star