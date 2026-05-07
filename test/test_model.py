from model import ResourceGameModel

m = ResourceGameModel(R=10, C1=4, C2=3, replenishment_dist={0:0.2, 2:0.6, 4:0.2})

s = 5
print("States (first 5):", m.states()[:5])
print("A(s):", m.actions_p1(s))
print("B(s):", m.actions_p2(s))
print("reward(s, a=3, b=2):", m.reward(s, 3, 2))
print("P(s'|s,a=3,b=2):", m.transition_probs(s, 3, 2))