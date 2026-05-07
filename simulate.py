import numpy as np
from typing import Dict, List, Tuple
from model import ResourceGameModel


def sample_from_probs(actions: List[int], probs: np.ndarray) -> int:
    """Sample an action from a discrete distribution."""
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    idx = np.random.choice(len(actions), p=probs)
    return actions[idx]


def sample_next_state(trans_probs: Dict[int, float]) -> int:
    """Sample next state s' from a dict {s': prob}."""
    states = list(trans_probs.keys())
    probs = np.array([trans_probs[s] for s in states], dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(states, p=probs))


def rollout(
    model: ResourceGameModel,
    pi1: Dict[int, np.ndarray],
    pi2: Dict[int, np.ndarray],
    s0: int,
    T: int = 30,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int], List[float], float]:
    """
    Simulate a trajectory using stationary policies.
    Returns:
        states: [s0, s1, ..., sT]
        actions1: [a0, ..., a_{T-1}]
        actions2: [b0, ..., b_{T-1}]
        rewards: [r0, ..., r_{T-1}]  (player 1 reward)
        discounted_return: sum_{t=0}^{T-1} gamma^t * r_t
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    s = int(s0)
    states = [s]
    actions1, actions2, rewards = [], [], []
    G = 0.0
    discount = 1.0

    for t in range(T):
        A = model.actions_p1(s)
        B = model.actions_p2(s)

        a = sample_from_probs(A, pi1[s])
        b = sample_from_probs(B, pi2[s])

        r = model.reward(s, a, b)
        trans = model.transition_probs(s, a, b)
        s_next = sample_next_state(trans)

        actions1.append(a)
        actions2.append(b)
        rewards.append(r)

        G += discount * r
        discount *= model.gamma

        s = s_next
        states.append(s)

    return states, actions1, actions2, rewards, G