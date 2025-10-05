## 1. Overview

**Reinforcement Learning** is the study of how agents learn to act optimally through **experience**—by interacting with an environment, observing the consequences of actions, and receiving **rewards**.

Unlike in MDP planning, in RL:
- The agent does **not know** the transition probabilities T(s, a, s′) or rewards R(s, a, s′).
- It must **learn by trial and error**.
- The goal is to learn an **optimal policy π*(s)** that maximizes expected cumulative reward.

---

## 2. The Agent–Environment Loop

At each time step _t_:

1. The agent observes **state** sₜ.
2. Chooses an **action** aₜ = π(sₜ).
3. Receives a **reward** rₜ₊₁ and the next **state** sₜ₊₁.
4. Updates its knowledge or policy based on experience.

Formally:  
sₜ —(aₜ)→ (rₜ₊₁, sₜ₊₁)

## 3. Two Main RL Settings
|Setting|Known Model?|Example|Algorithm|
|---|---|---|---|
|**Model-Based RL**|Yes (or learned model of T, R)|Simulated planning (e.g., Dyna-Q)|Value Iteration, Policy Iteration|
|**Model-Free RL**|No (learns directly from samples)|Robot learning to walk|Q-Learning, SARSA|
## 4. Passive vs. Active RL
- **Passive RL:**  
    The agent follows a **fixed policy** π and estimates its value function from experience.  
    (Evaluates how good π is.)
- **Active RL:**  
    The agent **learns both the policy and the values** by exploring the environment.

In both cases, the agent refines **state utilities** or **action-values** using sampled rewards.

## 5. Temporal Difference Learning
TD learning updates estimates of utilities using the **difference** between predicted and actual outcomes.

For a given experience (s, r, s′):

U(s) ← U(s) + α [ r + γ U(s′) − U(s) ]

where:
- α = learning rate (0 < α ≤ 1)
- r + γU(s′) is the _target_ (new information)
- [ r + γU(s′) − U(s) ] is the _temporal difference error_

This merges the ideas of **Monte Carlo sampling** (learning from experience) and **bootstrapping** (using previous estimates).

## 6. Q-Learning

The most famous **model-free** RL algorithm.

Instead of estimating utilities U(s), it learns **action-values** Q(s, a):  
Expected future reward from taking action _a_ in state _s_ and following the optimal policy thereafter.

**Update rule:**
Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]

---
### Pseudocode for Q-Learning

Q-LEARNING():  
 Initialize Q(s, a) arbitrarily for all states and actions  
 Repeat for each episode:  
  Initialize s  
  Repeat until s is terminal:  
   Choose action a using exploration policy (e.g., ε-greedy)  
   Execute a, observe r, s′  
   Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]  
   s ← s′

Returns the optimal action-value function Q*(s, a).

```python
import random
from collections import defaultdict

def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    env must define:
      - env.reset() -> state
      - env.step(action) -> next_state, reward, done
      - env.actions(state) -> list of possible actions
    """
    Q = defaultdict(lambda: defaultdict(float))

    for _ in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                a = random.choice(env.actions(s))
            else:
                a = max(env.actions(s), key=lambda a: Q[s][a])

            s_next, r, done = env.step(a)
            best_next = max(Q[s_next].values()) if Q[s_next] else 0.0

            # Q-learning update
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])

            s = s_next

    return Q
```

## 7. Exploration vs. Exploitation

The agent faces a critical trade-off:
- **Exploration:** try new actions to learn their outcomes.
- **Exploitation:** choose actions that currently seem best.
### ε-Greedy Strategy

With probability ε, choose a random action (explore).  
With probability 1−ε, choose the best-known action (exploit).

ε is often **decayed** over time (e.g., ε = 1 / episode_number).

---
## 8. Off-Policy vs. On-Policy Learning

- **Off-Policy (e.g., Q-Learning):**  
    Learns the optimal policy **independent** of the behavior policy (can explore randomly).
- **On-Policy (e.g., SARSA):**  
    Learns the value of the **current** behavior policy.  
    Update uses the **actual** action taken next, not the max future action.

**SARSA update:**  
Q(s, a) ← Q(s, a) + α [ r + γ Q(s′, a′) − Q(s, a) ]

---
## 9. Approximate Q-Learning

In large or continuous state spaces, storing Q(s, a) for every pair is infeasible.  
We instead use **features** f₁, f₂, …, fₙ and learn **weights** w₁, w₂, …, wₙ:

Q(s, a) ≈ w₁f₁(s, a) + w₂f₂(s, a) + … + wₙfₙ(s, a)

Update rule:
wᵢ ← wᵢ + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ] fᵢ(s, a)

```python
import numpy as np

class LinearQAgent:
    def __init__(self, num_features, alpha=0.01, gamma=0.9):
        self.w = np.zeros(num_features)
        self.alpha = alpha
        self.gamma = gamma

    def q_value(self, features):
        return np.dot(self.w, features)

    def update(self, features, reward, next_features, done):
        prediction = np.dot(self.w, features)
        target = reward if done else reward + self.gamma * np.dot(self.w, next_features)
        error = target - prediction
        self.w += self.alpha * error * features

```

## 10. Policy Search

Instead of approximating value functions, **Policy Search** methods directly tune policy parameters θ to maximize expected return J(θ).

Example: gradient ascent on policy performance  
θ ← θ + α ∇_θ J(θ)

Used in advanced algorithms like **REINFORCE**, **Actor–Critic**, and **PPO**.

---

## 11. Regret and Learning Performance

Even optimal RL algorithms incur mistakes during exploration.  
**Regret** measures the total reward difference between the agent and the optimal policy:

Regret(T) = ∑ₜ [ R*(t) − R_agent(t) ]

Goal: minimize regret growth (ideally sublinear in T).


### connections
- **From MDPs:** RL learns what MDP solvers assume known (T, R).
- **To Deep RL:** Approximate Q-learning becomes Deep Q-Networks (DQN).
- **To Psychology:** TD learning models dopamine-based reward prediction errors in the brain.

### glossary
- **Q(s, a):** Expected cumulative reward from (s, a).
- **Temporal Difference (TD):** Difference between successive predictions.
- **Learning Rate (α):** Step size for each update.
- **Discount Factor (γ):** Weight for future rewards.
- **Exploration Policy:** Determines how new states are sampled.

### takeaways
- RL learns through interaction, not supervision.
- Temporal Difference methods update estimates based on partial outcomes.
- Q-learning converges to optimal behavior with enough exploration and a decaying learning rate.
- Approximation and feature-based methods make RL scalable.
- Balancing exploration and exploitation is crucial for success.