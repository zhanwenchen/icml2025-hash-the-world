"""
toy_env_example.py
===================

This script demonstrates how to integrate the hashing‑based exploration agent
with a simple text environment. It constructs a tiny deterministic environment
using the `S` and `A` lists from our ICML paper, runs a few episodes with an
ε‑greedy policy, and prints the episode returns along with the learned action
clusters.

Run this script after installing the `mmh3` dependency:

```
pip install mmh3
python toy_env_example.py
```

The example is intentionally minimal and is meant only as a starting point
for integrating the agent into more complex LM‑RL environments.
"""

import random
from collections import defaultdict

from hashing_bucketers import HashingAgentIndex

class ToyTextEnv:
    """A tiny deterministic text environment.

    The environment has four states and four correct actions. The agent
    receives a reward of 1.0 only when it chooses the correct action at each
    step, otherwise 0.0. Episodes terminate after the last step.
    """

    def __init__(self, states, actions):
        assert len(states) == len(actions)
        self.S = list(states)
        self.A = list(actions)
        self.t = 0

    def reset(self):
        self.t = 0
        return self.S[self.t]

    def propose_actions(self, state_text):
        # For demonstration, we propose a small fixed set of actions including
        # the correct one. Real agents would use templates, KGs or LLMs.
        base = ["look around", "open door to kitchen", "go to kitchen"]
        correct = self.A[self.t]
        if correct not in base:
            base.append(correct)
        # deduplicate while preserving order
        return list(dict.fromkeys(base))

    def step(self, action_text):
        correct = self.A[self.t]
        reward = 1.0 if action_text == correct else 0.0
        done = (self.t == len(self.S) - 1)
        if not done:
            self.t += 1
        next_state = self.S[self.t]
        return next_state, reward, done, {}

def select_action(agent, state_text, candidates, eps=0.1):
    """ε‑greedy action selection using the agent's value function.

    The function peeks at Q‑values without updating buckets. With
    probability `eps` it chooses a random action for exploration.
    """
    if random.random() < eps:
        return random.choice(candidates)
    scores = []
    for a in candidates:
        # peek Q without side effects
        q = agent.value(state_text, a)
        scores.append((q, a))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]

def run_episode(agent, env, eps=0.1):
    """Run one episode and return the total reward."""
    state = env.reset()
    done = False
    # choose initial action
    candidates = env.propose_actions(state)
    action = select_action(agent, state, candidates, eps=eps)
    total_reward = 0.0
    while not done:
        next_state, ext_r, done, _ = env.step(action)
        total_reward += ext_r
        if not done:
            next_candidates = env.propose_actions(next_state)
            next_action = select_action(agent, next_state, next_candidates, eps=eps)
        else:
            next_action = None
        # update agent (SARSA) using current and next actions
        agent.update(state, action, ext_r,
                     next_state if not done else None,
                     next_action)
        state, action = next_state, next_action
    return total_reward

def main():
    # Our toy states and actions from the ICML paper example
    S = [
        "your task is to boil water ... change its state of matter",
        "this room is called the hallway ... a door to the kitchen ...",
        "the door is already open",
        "you move to the kitchen",
    ]
    A = ["look around", "open door to kitchen", "go to kitchen", "look around"]
    env = ToyTextEnv(S, A)
    agent = HashingAgentIndex(B_state=8192,
                              act_dist_thr=18,
                              act_min_absorb=5,
                              c_bonus=0.5,
                              gamma=0.99,
                              alpha=0.1)
    num_episodes = 5
    for ep in range(num_episodes):
        ret = run_episode(agent, env, eps=0.2)
        print(f"episode {ep}: return={ret}")
    # Inspect the learned buckets
    print("\nAction buckets:")
    for bid, info in agent.action_bucketer._registry.items():
        preview = list(info.members)[:5]
        print(f"  id={bid} count={info.count} members≈{preview}")

if __name__ == "__main__":
    main()
