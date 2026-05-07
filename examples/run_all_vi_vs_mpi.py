import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from model import ResourceGameModel
from shapley_vi import shapley_value_iteration, modified_policy_iteration
from simulate import rollout


def run_solver(name, solver_fn, model, s0, rollout_T=40, seed=0, **kwargs):
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = solver_fn(model, **kwargs)
    t1 = time.perf_counter()

    states, a1, a2, rewards, G = rollout(model, pi1, pi2, s0=s0, T=rollout_T, seed=seed)

    return {
        "solver": name,
        "runtime_sec": t1 - t0,
        "iters": len(deltas),
        "final_delta": float(deltas[-1]),
        "V_min": float(np.min(V)),
        "V_max": float(np.max(V)),
        "rollout_return": float(G),
    }


def format_row(r):
    return (f"{r['solver']:>4} | time={r['runtime_sec']:.3f}s | iters={r['iters']:>4} | "
            f"delta={r['final_delta']:.2e} | V_range=({r['V_min']:.3f},{r['V_max']:.3f}) | "
            f"rollout={r['rollout_return']:.3f}")


def run_case(case_name, model):
    print(f"\n=== {case_name} ===")
    s0 = model.R

    res_vi = run_solver(
        "VI",
        shapley_value_iteration,
        model,
        s0=s0,
        eps=1e-5,
        max_iter=5000,
        solver="ECOS"
    )

    res_mpi = run_solver(
        "MPI",
        modified_policy_iteration,
        model,
        s0=s0,
        eps=1e-5,
        max_iter=5000,
        solver="ECOS",
        eval_steps=5
    )

    print(format_row(res_vi))
    print(format_row(res_mpi))
    return res_vi, res_mpi


def main():
    out_dir = "figures/vi_vs_mpi"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vi_vs_mpi_summary.txt")

    results = []

    # Ex1: deterministic replenishment
    model1 = ResourceGameModel(R=20, C1=5, C2=4, gamma=0.95, replenishment_dist={2: 1.0})
    results += [("Ex1_deterministic",) + run_case("Ex1 (Deterministic)", model1)]

    # Ex2: stochastic replenishment
    model2 = ResourceGameModel(R=20, C1=5, C2=4, gamma=0.95, replenishment_dist={0: 0.2, 2: 0.6, 4: 0.2})
    results += [("Ex2_stochastic",) + run_case("Ex2 (Stochastic)", model2)]

    # Ex3: asymmetry
    model3 = ResourceGameModel(R=20, C1=6, C2=3, gamma=0.95, replenishment_dist={2: 1.0})
    results += [("Ex3_asymmetry",) + run_case("Ex3 (Asymmetry)", model3)]

    # Write summary table
    with open(out_path, "w") as f:
        f.write("Case\tSolver\truntime_sec\titers\tfinal_delta\tV_min\tV_max\trollout_return\n")
        for case, vi, mpi in results:
            for r in [vi, mpi]:
                f.write(f"{case}\t{r['solver']}\t{r['runtime_sec']:.6f}\t{r['iters']}\t{r['final_delta']:.3e}\t"
                        f"{r['V_min']:.6f}\t{r['V_max']:.6f}\t{r['rollout_return']:.6f}\n")

    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()