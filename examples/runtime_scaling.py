import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from model import ResourceGameModel
from shapley_vi import shapley_value_iteration


def run_one(R, C1=5, C2=4, gamma=0.95, replenishment_dist=None):
    model = ResourceGameModel(
        R=R,
        C1=C1,
        C2=C2,
        gamma=gamma,
        replenishment_dist=replenishment_dist or {2: 1.0},
    )
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = shapley_value_iteration(model, eps=1e-5, max_iter=5000, solver="ECOS")
    t1 = time.perf_counter()

    return {
        "R": R,
        "iters": len(deltas),
        "final_delta": float(deltas[-1]),
        "runtime_sec": float(t1 - t0),
        "V_min": float(np.min(V)),
        "V_max": float(np.max(V)),
    }


def main():
    out_dir = "figures/runtime_scaling"
    os.makedirs(out_dir, exist_ok=True)

    R_list = [10, 20, 30, 40, 60]  # 你可以按时间改少一点
    results = []

    print("=== Runtime scaling experiment ===")
    for R in R_list:
        res = run_one(R)
        results.append(res)
        print(res)

    # Save a simple CSV-like text table
    table_path = os.path.join(out_dir, "runtime_scaling_table.txt")
    with open(table_path, "w") as f:
        f.write("R\titers\tfinal_delta\truntime_sec\tV_min\tV_max\n")
        for r in results:
            f.write(f"{r['R']}\t{r['iters']}\t{r['final_delta']:.3e}\t{r['runtime_sec']:.4f}\t{r['V_min']:.4f}\t{r['V_max']:.4f}\n")

    # Plot runtime vs R
    plt.figure()
    plt.plot([r["R"] for r in results], [r["runtime_sec"] for r in results], marker="o")
    plt.xlabel("State size R (|S| = R+1)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime scaling vs state size")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_R.png"), dpi=200)

    # Plot iterations vs R
    plt.figure()
    plt.plot([r["R"] for r in results], [r["iters"] for r in results], marker="o")
    plt.xlabel("State size R (|S| = R+1)")
    plt.ylabel("Value iteration iterations")
    plt.title("Iteration scaling vs state size")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iters_vs_R.png"), dpi=200)

    plt.close("all")
    print(f"\nSaved figures and table to: {out_dir}")


if __name__ == "__main__":
    main()