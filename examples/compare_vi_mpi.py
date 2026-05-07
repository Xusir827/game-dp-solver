import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import ResourceGameModel
from shapley_vi import shapley_value_iteration, modified_policy_iteration


def run_vi(model):
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = shapley_value_iteration(model, eps=1e-5, max_iter=5000, solver="ECOS")
    t1 = time.perf_counter()
    return (t1 - t0), len(deltas), deltas[-1]


def run_mpi(model, eval_steps):
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = modified_policy_iteration(
        model, eps=1e-5, max_iter=5000, solver="ECOS", eval_steps=eval_steps
    )
    t1 = time.perf_counter()
    return (t1 - t0), len(deltas), deltas[-1]


def main():
    model = ResourceGameModel(
        R=20, C1=5, C2=4, gamma=0.95,
        replenishment_dist={0:0.2, 2:0.6, 4:0.2}
    )

    vi_time, vi_iters, vi_delta = run_vi(model)
    print("VI   time:", round(vi_time, 3), "iters:", vi_iters, "final:", vi_delta)

    for k in [1, 3, 5, 10]:
        mpi_time, mpi_iters, mpi_delta = run_mpi(model, eval_steps=k)
        print(f"MPI(eval_steps={k}) time:", round(mpi_time, 3), "iters:", mpi_iters, "final:", mpi_delta)


if __name__ == "__main__":
    main()