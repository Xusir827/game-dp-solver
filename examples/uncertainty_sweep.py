import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from model import ResourceGameModel
from shapley_vi import modified_policy_iteration


def dominant_actions(model, pi, player="P1"):
    acts = []
    for s in model.states():
        actions = model.actions_p1(s) if player == "P1" else model.actions_p2(s)
        probs = pi[s]
        acts.append(actions[int(np.argmax(probs))])
    return np.array(acts)


def run_one(label, replenishment_dist, gamma=0.95, eval_steps=5):
    model = ResourceGameModel(
        R=20,
        C1=5,
        C2=4,
        gamma=gamma,
        replenishment_dist=replenishment_dist
    )

    t0 = time.perf_counter()
    V, pi1, pi2, deltas = modified_policy_iteration(
        model, eps=1e-5, max_iter=5000, solver="ECOS", eval_steps=eval_steps
    )
    t1 = time.perf_counter()

    return model, {
        "label": label,
        "dist": replenishment_dist,
        "runtime": t1 - t0,
        "iters": len(deltas),
        "final_delta": float(deltas[-1]),
        "V": V,
        "a1": dominant_actions(model, pi1, player="P1"),
        "a2": dominant_actions(model, pi2, player="P2"),
    }


def main():
    out_dir = "figures/uncertainty_sweep"
    os.makedirs(out_dir, exist_ok=True)

    # All distributions have E[w]=2, but increasing variance.
    cases = [
        ("Low (deterministic)", {2: 1.0}),
        ("Medium", {0: 0.2, 2: 0.6, 4: 0.2}),
        ("High", {0: 0.4, 2: 0.2, 4: 0.4}),
    ]

    results = []
    models = []

    for label, dist in cases:
        model, res = run_one(label, dist)
        models.append(model)
        results.append(res)
        print(f"{label:>16} | time={res['runtime']:.3f}s | iters={res['iters']} | final={res['final_delta']:.2e} | dist={dist}")

    R = models[0].R
    states = np.arange(R + 1)

    # ---- Plot 1: V(s) curves ----
    plt.figure()
    for res in results:
        plt.plot(states, res["V"], marker="o", label=res["label"])
    plt.xlabel("State s")
    plt.ylabel("V(s)")
    plt.title("Uncertainty sweep (same mean replenishment): Value function V(s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "V_vs_state_uncertainty_sweep.png"), dpi=200)

    # ---- Plot 2: P1 dominant action vs state ----
    plt.figure()
    for res in results:
        plt.plot(states, res["a1"], marker="o", label=res["label"])
    plt.xlabel("State s")
    plt.ylabel("P1 dominant action (argmax)")
    plt.title("Uncertainty sweep: P1 dominant action vs state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "P1_action_vs_state_uncertainty_sweep.png"), dpi=200)

    # ---- Plot 3: P2 dominant action vs state ----
    plt.figure()
    for res in results:
        plt.plot(states, res["a2"], marker="o", label=res["label"])
    plt.xlabel("State s")
    plt.ylabel("P2 dominant action (argmax)")
    plt.title("Uncertainty sweep: P2 dominant action vs state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "P2_action_vs_state_uncertainty_sweep.png"), dpi=200)

    # Save text summary
    summary_path = os.path.join(out_dir, "uncertainty_sweep_summary.txt")
    with open(summary_path, "w") as f:
        f.write("label\truntime_sec\titers\tfinal_delta\tdist\n")
        for res in results:
            f.write(f"{res['label']}\t{res['runtime']:.6f}\t{res['iters']}\t{res['final_delta']:.3e}\t{res['dist']}\n")

    plt.close("all")
    print(f"\nSaved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()