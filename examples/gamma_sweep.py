import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from model import ResourceGameModel
from shapley_vi import shapley_value_iteration, modified_policy_iteration


def dominant_actions(model, pi, player="P1"):
    acts = []
    for s in model.states():
        actions = model.actions_p1(s) if player == "P1" else model.actions_p2(s)
        probs = pi[s]
        acts.append(actions[int(np.argmax(probs))])
    return np.array(acts)


def run_one(gamma, use_mpi=True, eval_steps=5):
    # Fix the environment so only gamma changes.
    model = ResourceGameModel(
        R=20,
        C1=5,
        C2=4,
        gamma=gamma,
        replenishment_dist={0: 0.2, 2: 0.6, 4: 0.2}  # choose stochastic as a representative system setting
    )

    t0 = time.perf_counter()
    if use_mpi:
        V, pi1, pi2, deltas = modified_policy_iteration(
            model, eps=1e-5, max_iter=5000, solver="ECOS", eval_steps=eval_steps
        )
        method = f"MPI(eval={eval_steps})"
    else:
        V, pi1, pi2, deltas = shapley_value_iteration(
            model, eps=1e-5, max_iter=5000, solver="ECOS"
        )
        method = "VI"
    t1 = time.perf_counter()

    res = {
        "gamma": gamma,
        "method": method,
        "runtime": t1 - t0,
        "iters": len(deltas),
        "final_delta": float(deltas[-1]),
        "V": V,
        "a1": dominant_actions(model, pi1, player="P1"),
        "a2": dominant_actions(model, pi2, player="P2"),
    }
    return model, res


def main():
    out_dir = "figures/gamma_sweep"
    os.makedirs(out_dir, exist_ok=True)

    gammas = [0.90, 0.95, 0.99]
    use_mpi = True
    eval_steps = 5

    models = []
    results = []

    for g in gammas:
        model, res = run_one(g, use_mpi=use_mpi, eval_steps=eval_steps)
        models.append(model)
        results.append(res)
        print(f"gamma={g:.2f} method={res['method']} time={res['runtime']:.3f}s iters={res['iters']} final={res['final_delta']:.2e}")

    # Sanity: all models share same R
    R = models[0].R
    states = np.arange(R + 1)

    # ---- Plot 1: V(s) curves ----
    plt.figure()
    for res in results:
        plt.plot(states, res["V"], marker="o", label=f"γ={res['gamma']:.2f}")
    plt.xlabel("State s")
    plt.ylabel("V(s)")
    plt.title("Gamma sweep: Value function V(s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "V_vs_state_gamma_sweep.png"), dpi=200)

    # ---- Plot 2: P1 dominant action vs state ----
    plt.figure()
    for res in results:
        plt.plot(states, res["a1"], marker="o", label=f"γ={res['gamma']:.2f}")
    plt.xlabel("State s")
    plt.ylabel("P1 dominant action (argmax)")
    plt.title("Gamma sweep: P1 dominant action vs state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "P1_action_vs_state_gamma_sweep.png"), dpi=200)

    # ---- Plot 3: P2 dominant action vs state ----
    plt.figure()
    for res in results:
        plt.plot(states, res["a2"], marker="o", label=f"γ={res['gamma']:.2f}")
    plt.xlabel("State s")
    plt.ylabel("P2 dominant action (argmax)")
    plt.title("Gamma sweep: P2 dominant action vs state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "P2_action_vs_state_gamma_sweep.png"), dpi=200)

    # Save a small text summary
    summary_path = os.path.join(out_dir, "gamma_sweep_summary.txt")
    with open(summary_path, "w") as f:
        f.write("gamma\tmethod\truntime_sec\titers\tfinal_delta\n")
        for res in results:
            f.write(f"{res['gamma']:.2f}\t{res['method']}\t{res['runtime']:.6f}\t{res['iters']}\t{res['final_delta']:.3e}\n")

    plt.close("all")
    print(f"Saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()