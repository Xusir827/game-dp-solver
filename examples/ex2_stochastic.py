import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
from model import ResourceGameModel
from shapley_vi import shapley_value_iteration
from simulate import rollout
import viz


def main():
    out_dir = "figures/ex2_stochastic"
    os.makedirs(out_dir, exist_ok=True)

    # Example 2: stochastic replenishment
    model = ResourceGameModel(
        R=20,
        C1=5,
        C2=4,
        gamma=0.95,
        replenishment_dist={0: 0.2, 2: 0.6, 4: 0.2},  # stochastic
    )

    V, pi1, pi2, deltas = shapley_value_iteration(model, eps=1e-5, max_iter=3000, solver="ECOS")

    print("=== Example 2 (Stochastic) ===")
    print("iters:", len(deltas), "final delta:", deltas[-1])
    print("V range:", (min(V), max(V)))

    # Save figures
    viz.plot_value_function(V, title="V(s) - Ex2 (Stochastic)",
                            save_path=os.path.join(out_dir, "V.png"))
    viz.plot_convergence(deltas, title="Convergence - Ex2 (Stochastic)",
                         save_path=os.path.join(out_dir, "convergence.png"))
    viz.plot_deterministic_policy_curve(model, pi1, player_name="P1",
                                        save_path=os.path.join(out_dir, "P1_action_vs_state.png"))
    viz.plot_deterministic_policy_curve(model, pi2, player_name="P2",
                                        save_path=os.path.join(out_dir, "P2_action_vs_state.png"))

    # Rollout
    s0 = model.R
    states, a1, a2, rewards, G = rollout(model, pi1, pi2, s0=s0, T=40, seed=0)
    print("rollout discounted return:", G)
    viz.plot_rollout(states, a1, a2, rewards, title="Rollout - Ex2", save_dir=out_dir)

    plt.close("all")


if __name__ == "__main__":
    main()