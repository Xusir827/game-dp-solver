from model import ResourceGameModel
from shapley_vi import shapley_value_iteration
from simulate import rollout
import viz
import matplotlib.pyplot as plt


def pick_informative_states(model, pi1, pi2, max_states=5, threshold=0.98):
    """
    Pick states where policies are not almost-pure.
    We select states where max(prob) < threshold for either player.
    """
    candidates = []
    for s in model.states():
        if pi1[s] is None or pi2[s] is None:
            continue
        if max(pi1[s]) < threshold or max(pi2[s]) < threshold:
            candidates.append(s)

    # Fallback if everything is (almost) pure
    if not candidates:
        # choose a few spread-out states
        candidates = [0, model.R // 3, 2 * model.R // 3, model.R]
        # remove duplicates and keep valid
        candidates = sorted(list(set([int(x) for x in candidates if 0 <= x <= model.R])))

    return candidates[:max_states]


def main():
    # ---- Example settings (demo) ----
    model = ResourceGameModel(
        R=10,
        C1=4,
        C2=3,
        gamma=0.95,
        # Deterministic replenishment for demo. (Example 2 will be stochastic.)
        replenishment_dist={0:0.2, 2:0.6, 4:0.2}
        # reward uses your current proportional-allocation version in model.py
        # (no need to set lambda_penalty for infeasible now)
    )

    # ---- Solve via Shapley value iteration ----
    V, pi1, pi2, deltas = shapley_value_iteration(
        model,
        eps=1e-5,
        max_iter=2000,
        solver="ECOS"
    )

    print("Converged iterations:", len(deltas))
    print("Final delta:", deltas[-1])
    print("V:", V)

    # ---- Plots: value & convergence ----
    viz.plot_value_function(V, title="V(s) - Demo")
    viz.plot_convergence(deltas, title="Convergence - Demo")

    # ---- Plot policies for informative states ----
    states_to_plot = pick_informative_states(model, pi1, pi2, max_states=5, threshold=0.98)
    print("States selected for policy plots:", states_to_plot)

    viz.plot_deterministic_policy_curve(model, pi1, player_name="P1")
    viz.plot_deterministic_policy_curve(model, pi2, player_name="P2")

    # ---- Rollout demo ----
    s0 = model.R
    states, a1, a2, rewards, G = rollout(model, pi1, pi2, s0=s0, T=30, seed=0)
    print(f"Discounted return from s0={s0}:", G)

    viz.plot_rollout(states, a1, a2, rewards, title=f"Rollout from s0={s0}")

    plt.show()


if __name__ == "__main__":
    main()