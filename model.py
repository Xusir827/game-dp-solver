import numpy as np
from typing import Dict, List, Callable, Optional


class ResourceGameModel:
    """
    A simple dynamic resource-competition model (zero-sum).

    State:
        s in {0, 1, ..., R}  (remaining capacity)

    Actions:
        Player 1 chooses a in {0..min(C1, s)}
        Player 2 chooses b in {0..min(C2, s)}

    Transition:
        s' = clip(s - a - b + w, 0, R)
        where w is replenishment (deterministic or stochastic)

    Payoff (to Player 1):
        r(s,a,b) = u(a) - u(b) - lambda_penalty * max(0, a+b-s)
        Player 2 payoff is -r.
    """

    def __init__(
        self,
        R: int = 20,
        C1: int = 5,
        C2: int = 5,
        gamma: float = 0.95,
        lambda_penalty: float = 10.0,
        utility: str = "sqrt",
        replenishment_dist: Optional[Dict[int, float]] = None,
    ):
        self.R = int(R)
        self.C1 = int(C1)
        self.C2 = int(C2)
        self.gamma = float(gamma)
        self.lambda_penalty = float(lambda_penalty)
        self.utility = utility

        # replenishment_dist: dict {w_value: prob}
        # If None -> deterministic replenishment w=2
        if replenishment_dist is None:
            self.replenishment_dist = {2: 1.0}
        else:
            # normalize just in case
            total = sum(replenishment_dist.values())
            self.replenishment_dist = {k: v / total for k, v in replenishment_dist.items()}

    def states(self) -> List[int]:
        return list(range(self.R + 1))

    def actions_p1(self, s: int) -> List[int]:
        return list(range(min(self.C1, s) + 1))

    def actions_p2(self, s: int) -> List[int]:
        return list(range(min(self.C2, s) + 1))

    def u(self, x: int) -> float:
        # simple concave utility
        if self.utility == "sqrt":
            return float(np.sqrt(x))
        elif self.utility == "log":
            return float(np.log1p(x))  # log(1+x)
        else:
            raise ValueError("utility must be 'sqrt' or 'log'")

    def reward(self, s: int, a: int, b: int) -> float:
        c = 0.1  # tune later
        if a + b <= 0:
            alloc1, alloc2 = 0.0, 0.0
        elif a + b <= s:
            alloc1, alloc2 = float(a), float(b)
        else:
            # proportional allocation when over-demand happens
            alloc1 = (a / (a + b)) * s
            alloc2 = (b / (a + b)) * s

        return (alloc1 - alloc2) - c * (alloc1 + alloc2)

    def transition_probs(self, s: int, a: int, b: int) -> Dict[int, float]:
        """
        Returns a dict mapping s' -> probability.
        """
        probs: Dict[int, float] = {}
        for w, pw in self.replenishment_dist.items():
            s_prime = s - a - b + w
            s_prime = max(0, min(self.R, s_prime))  # clip
            probs[s_prime] = probs.get(s_prime, 0.0) + pw

        # numerical normalize
        total = sum(probs.values())
        for k in list(probs.keys()):
            probs[k] /= total
        return probs