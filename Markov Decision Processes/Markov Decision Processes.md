## 1. Overview

A **Markov Decision Process (MDP)** models sequential decision-making under uncertainty.  
It provides the mathematical foundation for **planning** and **reinforcement learning**.

An MDP is defined by the tuple:  
**(S, A, T, R, γ)** where

- **S** = set of states
- **A** = set of actions
- **T(s, a, s′)** = transition probability: P(s′ | s, a)
- **R(s, a, s′)** = reward for taking action a in state s and landing in s′
- **γ** = discount factor, 0 ≤ γ < 1

The **Markov property** states:  
Future state depends only on the current state and action, **not** on the history.  
Mathematically: P(s_{t+1} | s_t, a_t, s_{t−1}, …) = P(s_{t+1} | s_t, a_t)

---

## 2. Policies and Utilities

A **policy** π(s) defines the action an agent takes in each state.  
Goal: find the **optimal policy** π* that maximizes **expected cumulative discounted reward**.

The **utility (value)** of a state under π is:
U_π(s) = $$ E[ R(s, a, s′) + γ R(s′, a′, s″) + γ² R(s″, a″, …) ] $$
The **optimal utility function** U*(s) satisfies the **Bellman Optimality Equation**:

U*(s) = $$ maxₐ ∑ₛ′ T(s, a, s′) [ R(s, a, s′) + γ U*(s′) ] $$
This recursive definition captures the essence of dynamic programming.

---

## 3. Example: Gridworld

A robot moves in a 2D grid with obstacles, stochastic movement, and rewards:

- Actions: {Up, Down, Left, Right}
- Transition probabilities:
    - 80%: goes intended direction
    - 10%: slips left
    - 10%: slips right
- Rewards:
    - +10 for goal cell
    - -10 for pit
    - small negative “living cost” (-0.04) for each step

Goal: maximize long-term expected reward by planning an optimal policy.

---
## 4. Utilities of Sequences and Discounting

We prefer **earlier** rewards over **later** ones.  
Thus, the **discount factor** γ controls preference for immediacy.

If rewards are r₁, r₂, r₃, … then total utility is:

U = r₁ + γr₂ + γ²r₃ + …

If γ = 0.9, future rewards are worth 90% of immediate ones.  
If γ = 1, all rewards are equally important (but can cause divergence in infinite tasks).

---

## 5. Value Iteration Algorithm

**Goal:** Compute U*(s) for all states.  
Start with U₀(s) = 0 and iteratively apply the Bellman update until convergence.


```scss
VALUE-ITERATION(MDP, ε):  
	initialize U(s) = 0 for all s  
	repeat:  
		Δ ← 0  
		for each state s:  
			U′(s) ← maxₐ ∑ₛ′ T(s, a, s′)[ R(s, a, s′) + γ U(s′) ]  
			Δ ← max(Δ, |U′(s) − U(s)|)  
			U ← U′  
	until Δ < ε (1 − γ) / γ  
	return U
```


Properties:

- Converges to optimal utilities U*
- Then derive policy π*(s) = $$ argmaxₐ ∑ₛ′ T(s, a, s′)[R(s, a, s′) + γU*(s′)] $$

## 6. Python
```python
def value_iteration(states, actions, transitions, rewards, gamma=0.9, epsilon=1e-4):
    U = {s: 0.0 for s in states}

    while True:
        delta = 0
        new_U = {}
        for s in states:
            if s not in actions:
                new_U[s] = rewards.get(s, 0)
                continue
            q_values = []
            for a in actions[s]:
                q = 0
                for (s2, p) in transitions[s][a]:
                    r = rewards.get((s, a, s2), rewards.get(s, 0))
                    q += p * (r + gamma * U[s2])
                q_values.append(q)
            new_U[s] = max(q_values)
            delta = max(delta, abs(new_U[s] - U[s]))
        U = new_U
        if delta < epsilon * (1 - gamma) / gamma:
            break
    return U

```


## 7. Policy Iteration
Value Iteration updates utilities for all actions every step.  
**Policy Iteration** alternates between:

1. **Policy Evaluation**: compute utilities for a fixed policy.
2. **Policy Improvement**: update policy using one-step lookahead.

```scss
POLICY-ITERATION(MDP):  
	initialize π(s) arbitrarily  
	repeat:  
	U ← POLICY-EVALUATION(π)  
	unchanged ← True  
	for each state s:  
	best_action ← argmaxₐ ∑ₛ′ T(s, a, s′)[R(s, a, s′) + γU(s′)]  
	if best_action ≠ π(s):  
		π(s) ← best_action  
		unchanged ← False  
	until unchanged  
	return π
```

## 8. Python Implementation - Policy Iteration
```python
def policy_iteration(states, actions, transitions, rewards, gamma=0.9, epsilon=1e-4):
    import copy
    # Initialize arbitrary policy
    policy = {s: list(actions[s])[0] for s in states if s in actions}
    U = {s: 0.0 for s in states}

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            new_U = copy.deepcopy(U)
            for s in states:
                if s not in actions:
                    continue
                a = policy[s]
                new_U[s] = sum(p * (rewards.get((s, a, s2), 0) + gamma * U[s2])
                               for s2, p in transitions[s][a])
                delta = max(delta, abs(new_U[s] - U[s]))
            U = new_U
            if delta < epsilon:
                break

        # Policy Improvement
        stable = True
        for s in states:
            if s not in actions:
                continue
            best_a = max(actions[s], key=lambda a: sum(
                p * (rewards.get((s, a, s2), 0) + gamma * U[s2])
                for s2, p in transitions[s][a]))
            if best_a != policy[s]:
                policy[s] = best_a
                stable = False
        if stable:
            return policy, U

```


## 9. Comparison: Value Iteration vs Policy Iteration
|Property|Value Iteration|Policy Iteration|
|---|---|---|
|Updates|Full Bellman backups each step|Alternates evaluation/improvement|
|Convergence|Usually slower but simpler|Fewer outer iterations|
|Implementation|Easier|More stable|
|Typical Use|Small MDPs, visualization|Larger state spaces, more efficient|
## 10. Time-Limited Value Iteration
If computation time is limited, run Value Iteration for fixed steps (depth-limited planning).  
Each iteration approximates expected utilities up to horizon k.

Uₖ₊₁(s) = maxₐ ∑ₛ′ T(s, a, s′)[R(s, a, s′) + γUₖ(s′)]

As k → ∞, values converge to U*.

## 11. Exploration of Parameters
- **γ (discount)**: higher values (close to 1) make agents more farsighted.
- **Reward shaping**: adding small step penalties can stabilize learning.
- **Noise**: models uncertainty in transitions; high noise encourages safer policies.

## 12. Example Outputs (Gridworld)

Optimal utilities might look like:
+---------------------------------+  
| +1.0 | +0.8 | +0.6 | +1.0 |  
| +0.7 | +0.5 | +0.4 | -1.0 |  
| +0.4 | +0.3 | +0.2 | +0.0 |  
+---------------------------------+

Policy (arrows):  
→ → ↑ ↑  
→ → ↑ ×  
↑ ↑ → •

### connections
- **Expectimax:** MDPs are like Expectimax trees with known transition probabilities.
- **Reinforcement Learning:** If T and R are unknown, we must estimate them via experience.
- **Dynamic Programming:** Both algorithms (Value Iteration and Policy Iteration) use Bellman updates.
- **Planning:** MDPs generalize deterministic planning to stochastic environments.

### glossary
- **MDP:** Model describing decision-making under uncertainty.
- **Policy:** Mapping from states to actions.
- **Value Function:** Expected return from a state.
- **Bellman Equation:** Recursive relation defining optimal value.
- **Discount Factor (γ):** Measures preference for immediacy.
### takeaways
- MDPs formalize sequential decision-making under uncertainty.
- Bellman equations define relationships between state values.
- Value Iteration and Policy Iteration compute optimal policies through repeated backups.
- Discounting ensures convergence and models time preference.
- MDPs form the foundation for Reinforcement Learning.