import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_value_function(V, title="Value Function V(s)", save_path=None):
    s = np.arange(len(V))
    plt.figure()
    plt.plot(s, V, marker="o")
    plt.xlabel("State s")
    plt.ylabel("V(s)")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)


def plot_convergence(deltas, title="Convergence: ||V_{k+1}-V_k||_inf", save_path=None):
    plt.figure()
    plt.plot(np.arange(1, len(deltas) + 1), deltas, marker="o")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Infinity norm difference")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)


def plot_policy_bars(actions, probs, title, save_path=None):
    plt.figure()
    plt.bar([str(a) for a in actions], probs)
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)


def plot_deterministic_policy_curve(model, pi, player_name="P1", save_path=None):
    """
    Plot the dominant (argmax) action for each state.
    Works even if policies are pure (still informative).
    """
    states = model.states()
    chosen = []
    for s in states:
        actions = model.actions_p1(s) if player_name == "P1" else model.actions_p2(s)
        probs = pi[s]
        a_star = actions[int(np.argmax(probs))]
        chosen.append(a_star)

    plt.figure()
    plt.plot(states, chosen, marker="o")
    plt.xlabel("State s")
    plt.ylabel("Chosen action (argmax)")
    plt.title(f"{player_name} dominant action vs state")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)


def plot_rollout(states, actions1, actions2, rewards, title="Rollout", save_dir=None):
    """
    If save_dir is provided, saves 3 figures:
      rollout_state.png, rollout_actions.png, rollout_reward.png
    """
    T = len(rewards)

    # state
    plt.figure()
    plt.plot(range(T + 1), states, marker="o")
    plt.xlabel("t")
    plt.ylabel("state s_t")
    plt.title(title + " (state)")
    plt.tight_layout()
    if save_dir:
        ensure_dir(save_dir)
        plt.savefig(os.path.join(save_dir, "rollout_state.png"), dpi=200)

    # actions
    plt.figure()
    plt.plot(range(T), actions1, marker="o", label="a_t (P1)")
    plt.plot(range(T), actions2, marker="o", label="b_t (P2)")
    plt.xlabel("t")
    plt.ylabel("action")
    plt.title(title + " (actions)")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "rollout_actions.png"), dpi=200)

    # rewards
    plt.figure()
    plt.plot(range(T), rewards, marker="o")
    plt.xlabel("t")
    plt.ylabel("reward r_t (P1)")
    plt.title(title + " (reward)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "rollout_reward.png"), dpi=200)