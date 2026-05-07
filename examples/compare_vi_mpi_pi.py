import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import ResourceGameModel
from shapley_vi import shapley_value_iteration, modified_policy_iteration, policy_iteration_exact


def time_run(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    V, pi1, pi2, deltas = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return name, (t1 - t0), len(deltas), deltas[-1], (float(min(V)), float(max(V)))


def main():
    model = ResourceGameModel(
        R=20, C1=5, C2=4, gamma=0.95,
        replenishment_dist={0:0.2, 2:0.6, 4:0.2}
    )

    rows = []
    rows.append(time_run("VI", shapley_value_iteration, model, eps=1e-5, max_iter=5000, solver="ECOS"))
    rows.append(time_run("MPI(eval=5)", modified_policy_iteration, model, eps=1e-5, max_iter=5000, solver="ECOS", eval_steps=5))
    rows.append(time_run("PI-exact", policy_iteration_exact, model, eps=1e-5, max_iter=200, solver="ECOS"))

    for name, t, iters, d, vr in rows:
        print(f"{name:>10} | time={t:.3f}s | iters={iters:>4} | final={d:.2e} | V_range=({vr[0]:.3f},{vr[1]:.3f})")


if __name__ == "__main__":
    main()