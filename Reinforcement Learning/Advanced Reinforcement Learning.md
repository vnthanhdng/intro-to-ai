## 1. Motivation

In real-world environments, the **state** and **action spaces** can be huge or continuous — making it impossible to store a table of Q(s, a) values.

**Approximate Reinforcement Learning** replaces discrete Q-tables with **parameterized functions** that generalize across states.

Example:  
Instead of storing Q[(position, velocity, action)],  
learn Q(s, a) ≈ **f₍w₎(s, a)** = w₁f₁(s, a) + w₂f₂(s, a) + … + wₙfₙ(s, a)

where:

- fᵢ(s, a) = feature i (numeric descriptor)
- wᵢ = learned weight
---

## 2. Linear Approximate Q-Learning

We approximate Q(s, a) using a **linear combination of features**.

**Prediction:**  
Q(s, a) = w · f(s, a)

**Update Rule:**  
w ← w + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ] f(s, a)

This is a **stochastic gradient descent (SGD)** update minimizing squared TD error.

### Pseudocode — Linear Approximate Q-Learning
```scss
APPROXIMATE-Q-LEARNING():
    initialize weights w randomly
    repeat (for each episode):
        initialize state s
        repeat (for each step in episode):
            choose action a using ε-greedy policy based on Q(s, a)
            take action a, observe reward r and next state s′
            δ ← r + γ maxₐ′ Q(s′, a′; w) − Q(s, a; w)
            w ← w + α δ f(s, a)
            s ← s′

```

### Python Implementation
```python
import numpy as np
import random

class ApproxQLearner:
    def __init__(self, feature_extractor, alpha=0.05, gamma=0.9, epsilon=0.1):
        self.weights = np.zeros(feature_extractor.num_features)
        self.extract = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def q_value(self, state, action):
        features = self.extract.get_features(state, action)
        return np.dot(self.weights, features)

    def choose_action(self, env, state):
        actions = env.actions(state)
        if random.random() < self.epsilon:
            return random.choice(actions)
        values = [self.q_value(state, a) for a in actions]
        return actions[int(np.argmax(values))]

    def update(self, state, action, reward, next_state, done, env):
        features = self.extract.get_features(state, action)
        next_actions = env.actions(next_state)
        max_next_q = max([self.q_value(next_state, a) for a in next_actions]) if not done else 0
        target = reward + self.gamma * max_next_q
        td_error = target - self.q_value(state, action)
        self.weights += self.alpha * td_error * features

```

## 3. Exploration Strategies
A well-designed RL agent must balance exploration and exploitation intelligently — especially in continuous or complex spaces.
### ε-Greedy (basic)
Choose random action with probability ε.
### Decayed ε
ε = ε₀ × decay_rate^t or ε = 1 / (1 + episode_number)  
→ gradually reduces exploration over time.
### Softmax (Boltzmann Exploration)
Choose actions probabilistically, favoring higher-Q actions:
P(a|s) = exp(Q(s, a)/τ) / ∑ₐ′ exp(Q(s, a′)/τ)
τ (temperature) controls randomness:
- High τ → more exploration
- Low τ → more exploitation

### Exploration Function (used in Active RL)
F(u, n) = u + k / (1 + n)  
→ adds bonus for less-visited states or actions.

## 4. Regret Minimization

**Regret** measures how much reward was lost due to not acting optimally.
Regret(T) = ∑ₜ [ V*(sₜ) − Q(sₜ, aₜ) ]

An agent’s learning objective is to keep **regret sublinear** (grows slower than T).  
This means that as time goes on, average performance approaches optimal.

Exploration methods such as **Upper Confidence Bounds (UCB)** or **Thompson Sampling** can explicitly control regret.

---

## 5. Convergence and Stability

For tabular Q-learning:
- Converges to Q* if all state-action pairs are visited infinitely often,  
    and α decreases appropriately (e.g., αₜ → 0).

For approximate Q-learning:
- May not converge exactly (especially with nonlinear function approximators like neural nets).
- Requires stable function updates or experience replay (used in Deep Q-Networks).

---

## 6. From Linear to Deep Q-Learning (DQN)

**Deep Q-Networks (DQN)** replace hand-crafted features with neural networks:  
Q(s, a; θ) ≈ neural_net(s)[a]

Key innovations:
1. **Experience Replay:** store past transitions (s, a, r, s′) and sample randomly to break correlation.
2. **Target Network:** use a separate copy of the network for stable targets.
3. **Batch Training:** perform gradient descent on minibatches of experiences.

Update:  
θ ← θ + α (r + γ maxₐ′ Q(s′, a′; θ⁻) − Q(s, a; θ)) ∇_θ Q(s, a; θ)

These methods enabled breakthroughs in Atari and Go.

---

## 7. Policy Search Methods

Instead of learning value functions, **policy-based methods** directly optimize the policy parameters θ.
### Policy Gradient Objective
J(θ) = E_πθ [ ∑ₜ γᵗ rₜ ]
Update rule (stochastic gradient ascent):  
θ ← θ + α ∇_θ J(θ)

### REINFORCE Algorithm (Monte Carlo Policy Gradient)
1. Run an episode using current policy πθ(a|s)
2. For each time step t:
    - Compute return Gₜ = ∑ₖ γᵏ rₜ₊ₖ
    - θ ← θ + α Gₜ ∇_θ log πθ(aₜ | sₜ)

This increases the probability of actions that lead to higher returns.

---

### Actor–Critic Methods

Combine **policy learning (actor)** and **value estimation (critic):**
- Actor updates policy parameters using ∇_θ log πθ(a|s) × TD-error.
- Critic estimates value function V(s) using TD learning.

This hybrid method improves stability and sample efficiency.

## 8. Summary of Algo Families
|Category|Learns What|Model Required?|Examples|
|---|---|---|---|
|**Model-Free, Value-Based**|Q(s, a)|No|Q-learning, SARSA|
|**Model-Free, Policy-Based**|π(a|s)|No|
|**Model-Based**|T(s, a, s′), R(s, a, s′)|Yes (learn or assume)|Dyna-Q, Planning|
|**Approximation**|f(s, a; w)|No|Linear Q, DQN|
|**Hybrid**|Both π and V|No|Actor–Critic, PPO, A3C|
