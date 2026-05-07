import numpy as np
from matrix_game_lp import solve_zero_sum_matrix_game


def pretty(x):
    return np.round(x, 4)


# ---------------------------
# Test 1: Rock-Paper-Scissors
# The equilibrium is uniform: (1/3, 1/3, 1/3), value = 0
# ---------------------------
M_rps = np.array([
    [0, -1,  1],
    [1,  0, -1],
    [-1, 1,  0]
], dtype=float)

v, p, q = solve_zero_sum_matrix_game(M_rps)
print("RPS value:", v)
print("RPS p:", pretty(p), "sum:", p.sum())
print("RPS q:", pretty(q), "sum:", q.sum())
print()


# ---------------------------
# Test 2: Matching Pennies
# Equilibrium is uniform: (0.5, 0.5), value = 0
# ---------------------------
M_mp = np.array([
    [ 1, -1],
    [-1,  1]
], dtype=float)

v, p, q = solve_zero_sum_matrix_game(M_mp)
print("MP value:", v)
print("MP p:", pretty(p), "sum:", p.sum())
print("MP q:", pretty(q), "sum:", q.sum())