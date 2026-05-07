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

def policy_evaluation_step(
    model,
    V: np.ndarray,
    s: int,
    pi1_s: np.ndarray,
    pi2_s: np.ndarray
) -> float:
    """
    One Bellman evaluation backup under FIXED policies (pi1, pi2), no LP.

    V_eval(s) = E_{a~pi1, b~pi2} [ r(s,a,b) + gamma * E_{s'} V(s') ]
    """
    A = model.actions_p1(s)
    B = model.actions_p2(s)

    val = 0.0
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            # (If you removed infeasible penalty earlier, no need to check a+b>s here.
            # If your reward handles over-demand via proportional allocation, it is always defined.)
            r = model.reward(s, a, b)
            trans = model.transition_probs(s, a, b)

            exp_future = 0.0
            for sp, p in trans.items():
                exp_future += p * V[sp]

            val += pi1_s[i] * pi2_s[j] * (r + model.gamma * exp_future)

    return val


def modified_policy_iteration(
    model,
    eps: float = 1e-5,
    max_iter: int = 2000,
    solver: Optional[str] = "ECOS",
    eval_steps: int = 5,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], List[float]]:
    """
    Modified Policy Iteration (MPI) for zero-sum stochastic games.

    Each outer iteration does:
      (1) Policy improvement: for each state solve the minimax matrix game via LP to get policies.
      (2) Partial policy evaluation: perform eval_steps sweeps using FIXED policies (no LP).

    Returns:
        V, pi1, pi2, deltas (same format as shapley_value_iteration)
    """
    states = model.states()
    V = np.zeros(model.R + 1, dtype=float)

    pi1: Dict[int, np.ndarray] = {s: None for s in states}
    pi2: Dict[int, np.ndarray] = {s: None for s in states}
    deltas: List[float] = []

    for k in range(max_iter):
        # ---- (1) Policy improvement (LP per state) ----
        for s in states:
            M_s = build_payoff_matrix(model, V, s)  # uses current V
            v, p, q = solve_zero_sum_matrix_game(M_s, solver=solver)
            pi1[s] = p
            pi2[s] = q

        # ---- (2) Partial policy evaluation (no LP) ----
        V_new = V.copy()
        for _ in range(eval_steps):
            V_tmp = np.zeros_like(V_new)
            for s in states:
                V_tmp[s] = policy_evaluation_step(model, V_new, s, pi1[s], pi2[s])
            V_new = V_tmp

        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new

        if delta <= eps:
            break

    return V, pi1, pi2, deltas

def build_expected_reward_and_transition(model, pi1: Dict[int, np.ndarray], pi2: Dict[int, np.ndarray]):
    """
    Given fixed stationary policies pi1, pi2, build:
      r_pi: shape (|S|,)
      P_pi: shape (|S|, |S|)
    where
      r_pi[s] = E_{a,b}[ r(s,a,b) ]
      P_pi[s,sp] = E_{a,b}[ P(sp|s,a,b) ]
    """
    states = model.states()
    nS = len(states)
    r_pi = np.zeros(nS, dtype=float)
    P_pi = np.zeros((nS, nS), dtype=float)

    for s in states:
        A = model.actions_p1(s)
        B = model.actions_p2(s)
        p = pi1[s]
        q = pi2[s]

        # expected reward and transition under fixed policies
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                w = p[i] * q[j]
                if w == 0:
                    continue

                r_pi[s] += w * model.reward(s, a, b)

                trans = model.transition_probs(s, a, b)  # dict {sp: prob}
                for sp, prob in trans.items():
                    P_pi[s, sp] += w * prob

        # numerical normalize row if tiny drift (should already sum to 1)
        row_sum = P_pi[s, :].sum()
        if row_sum > 0:
            P_pi[s, :] /= row_sum

    return r_pi, P_pi


def exact_policy_evaluation(model, pi1: Dict[int, np.ndarray], pi2: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Solve (I - gamma * P_pi) V = r_pi exactly for V under fixed policies.
    """
    r_pi, P_pi = build_expected_reward_and_transition(model, pi1, pi2)
    nS = len(r_pi)
    A = np.eye(nS) - model.gamma * P_pi
    V = np.linalg.solve(A, r_pi)
    return V


def policy_iteration_exact(
    model,
    eps: float = 1e-5,
    max_iter: int = 200,
    solver: Optional[str] = "ECOS",
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], List[float]]:
    """
    Policy Iteration with exact (linear-system) policy evaluation.

    Outer loop:
      1) Policy improvement: for each state solve minimax matrix game using current V to get new policies.
      2) Exact policy evaluation: solve (I - gamma P_pi) V = r_pi.
      3) Stop when ||V_new - V_old||_inf <= eps.

    Returns: V, pi1, pi2, deltas
    """
    states = model.states()
    V = np.zeros(model.R + 1, dtype=float)

    # initialize with some policies (e.g., uniform)
    pi1: Dict[int, np.ndarray] = {}
    pi2: Dict[int, np.ndarray] = {}
    for s in states:
        A = model.actions_p1(s)
        B = model.actions_p2(s)
        pi1[s] = np.ones(len(A)) / len(A)
        pi2[s] = np.ones(len(B)) / len(B)

    deltas: List[float] = []

    for k in range(max_iter):
        V_old = V.copy()

        # (1) Policy improvement (LP per state, using current V)
        for s in states:
            M_s = build_payoff_matrix(model, V, s)   # r + gamma E[V(s')]
            v, p, q = solve_zero_sum_matrix_game(M_s, solver=solver)
            pi1[s] = p
            pi2[s] = q

        # (2) Exact evaluation under fixed policies
        V = exact_policy_evaluation(model, pi1, pi2)

        delta = float(np.max(np.abs(V - V_old)))
        deltas.append(delta)
        if delta <= eps:
            break

    return V, pi1, pi2, deltas