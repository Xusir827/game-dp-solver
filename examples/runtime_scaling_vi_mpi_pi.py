import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from model import ResourceGameModel
from shapley_vi import shapley_value_iteration, modified_policy_iteration, policy_iteration_exact


def time_run(name, fn, model, **kwargs):
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = fn(model, **kwargs)
    t1 = time.perf_counter()
    return {
        "method": name,
        "time": float(t1 - t0),
        "iters": int(len(deltas)),
        "final_delta": float(deltas[-1]),
        "V_min": float(np.min(V)),
        "V_max": float(np.max(V)),
    }


def run_for_R(R, gamma=0.95, case="stochastic", eval_steps=5):
    # Keep everything fixed except R
    if case == "deterministic":
        dist = {2: 1.0}
    else:
        dist = {0: 0.2, 2: 0.6, 4: 0.2}

    model = ResourceGameModel(
        R=R, C1=5, C2=4, gamma=gamma,
        replenishment_dist=dist
    )

    res = []
    res.append(time_run("VI", shapley_value_iteration, model,
                        eps=1e-5, max_iter=5000, solver="ECOS"))
    res.append(time_run(f"MPI(eval={eval_steps})", modified_policy_iteration, model,
                        eps=1e-5, max_iter=5000, solver="ECOS", eval_steps=eval_steps))
    # PI-exact usually needs far fewer outer iters; max_iter can be small
    res.append(time_run("PI-exact", policy_iteration_exact, model,
                        eps=1e-5, max_iter=200, solver="ECOS"))

    return res


def main():
    out_dir = "figures/runtime_scaling_vi_mpi_pi"
    os.makedirs(out_dir, exist_ok=True)

    R_list = [10, 20, 30, 40, 60]     # 你可以改成更大，但 PI-exact 线性方程会更重
    gamma = 0.95
    case = "stochastic"               # 也可以改成 deterministic
    eval_steps = 5

    # Store results in dict: method -> list of (R, time, iters)
    methods = ["VI", f"MPI(eval={eval_steps})", "PI-exact"]
    time_map = {m: [] for m in methods}
    iter_map = {m: [] for m in methods}
    delta_map = {m: [] for m in methods}

    lines = []
    lines.append("R\tmethod\ttime_sec\titers\tfinal_delta\tV_min\tV_max\n")

    print("=== Runtime scaling: VI vs MPI vs PI-exact ===")
    for R in R_list:
        results = run_for_R(R, gamma=gamma, case=case, eval_steps=eval_steps)
        for r in results:
            m = r["method"]
            time_map[m].append((R, r["time"]))
            iter_map[m].append((R, r["iters"]))
            delta_map[m].append((R, r["final_delta"]))
            lines.append(f"{R}\t{m}\t{r['time']:.6f}\t{r['iters']}\t{r['final_delta']:.3e}\t{r['V_min']:.6f}\t{r['V_max']:.6f}\n")
        print(f"R={R}: " + " | ".join([f"{r['method']} {r['time']:.3f}s ({r['iters']} iters)" for r in results]))

    # Save table
    table_path = os.path.join(out_dir, "scaling_summary.txt")
    with open(table_path, "w") as f:
        f.writelines(lines)

    # Plot runtime vs R
    plt.figure()
    for m in methods:
        xs = [x for x, _ in time_map[m]]
        ys = [y for _, y in time_map[m]]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("State size R (|S| = R+1)")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime scaling ({case}, γ={gamma})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_R_vi_mpi_pi.png"), dpi=200)

    # Plot iterations vs R
    plt.figure()
    for m in methods:
        xs = [x for x, _ in iter_map[m]]
        ys = [y for _, y in iter_map[m]]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("State size R (|S| = R+1)")
    plt.ylabel("Outer iterations")
    plt.title(f"Iteration scaling ({case}, γ={gamma})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iters_vs_R_vi_mpi_pi.png"), dpi=200)

    plt.close("all")
    print(f"\nSaved plots and table to: {out_dir}")
    print(f"- {table_path}")
    print(f"- runtime_vs_R_vi_mpi_pi.png")
    print(f"- iters_vs_R_vi_mpi_pi.png")


if __name__ == "__main__":
    main()