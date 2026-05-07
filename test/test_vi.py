from model import ResourceGameModel
from shapley_vi import shapley_value_iteration

# Start with a small model to ensure it runs fast
model = ResourceGameModel(
    R=10,
    C1=4,
    C2=3,
    gamma=0.95,
    lambda_penalty=10.0,
    utility="sqrt",
    replenishment_dist={2: 1.0}  # deterministic
)

V, pi1, pi2, deltas = shapley_value_iteration(model, eps=1e-5, max_iter=2000,solver="ECOS")

print("Converged in iterations:", len(deltas))
print("Final delta:", deltas[-1])
print("V:", V)

# Print policies for a few states
for s in [0, 2, 5, 10]:
    print(f"\nState s={s}")
    print("A(s):", model.actions_p1(s), "pi1:", pi1[s])
    print("B(s):", model.actions_p2(s), "pi2:", pi2[s])