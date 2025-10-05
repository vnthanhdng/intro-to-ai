## 1. Overview
While Minimax assumes a perfectly adversarial opponent who minimizes your score, many real-world situations involve randomness -- dice rolls, shuffled decks, or unpredictable agent behavior.

Expectimax generalizes Minimax to handle stochastic outcomes by introducing chance nodes instead of strict Min nodes.

Examples:
- Backgammon: dice outcomes affect legal moves.
- Pacman with random ghost motion: ghosts move according to probabilities.
- Slot machines, card games, robot sensors: all include uncertainty.

Goal: maximize the expected utility, not the guaranteed minimum.

## 2. Expectimax Search Tree
A game tree now has three types of nodes:

| Node Type    | Role                           | Function                                     |
| ------------ | ------------------------------ | -------------------------------------------- |
| **MAX**      | Decision point for the agent   | Chooses the action maximizing expected value |
| **CHANCE**   | Represents stochastic outcomes | Computes weighted average of child utilities |
| **TERMINAL** | End states                     | Contain utility (numeric payoff)             |

## 3. The Expectimax Algorithm
The key difference from Minimax is how chance nodes are evaluated: instead of a min, we compute an expected value.

Mathematically:
Expectimax(s) =
- if s is terminal:  
     U(s)
- if s is a MAX node:  
     max over a ∈ Actions(s) of ∑ P(s′ | s, a) × Expectimax(s′)
- if s is a CHANCE node:  
     ∑ P(outcome) × Expectimax(successor)

**Intuition:**  
At each decision, the agent maximizes **expected** utility, assuming randomness is known but not adversarial.

## 4. Expectimax Pseudocode
```scss
EXPECTIMAX(state):

 if TERMINAL(state):  
  return UTILITY(state)

 if IS-MAX-NODE(state):  
  return max over actions [ EXPECTIMAX(RESULT(state, action)) ]

 if IS-CHANCE-NODE(state):  
  return ∑ₒ P(o) × EXPECTIMAX(RESULT(state, o))
```

## 5. Example -- Dice Game
Suppose rolling a die determines reward:
- 1 → −10
- 2–5 → +0
- 6 → +20

Expected value = (1/6)(−10) + (4/6)(0) + (1/6)(20) = +1.67
Even though there’s a possible −10, the expected utility is positive, so a rational expectimax agent would play.

## 6. Expectimax vs. Minimax
|Property|Minimax|Expectimax|
|---|---|---|
|Opponent Type|Perfectly adversarial|Random/stochastic|
|Node Type|MAX, MIN|MAX, CHANCE|
|Node Evaluation|max/min of children|expected value of children|
|Risk Attitude|Pessimistic (worst-case)|Average-case|
|Result|Optimal vs best adversary|Optimal vs random process|
In Backgammon, using Minimax would assume the dice are trying to hurt you -- an unrealistic model. Expectimax correctly averages over possible rolls.

## 7. Python
Below is a basic Expectimax function assuming a `Game` object with:
- `actions(state)`
- `result(state, action)`
- `terminal_test(state)`
- `utility(state)`
- `chance_outcomes(state)` returning list of (probability, next_state)

```python
def expectimax(state, game, depth_limit=4):
    if game.terminal_test(state) or depth_limit == 0:
        return game.utility(state)

    player = game.to_move(state)

    if player == "MAX":
        return max(expectimax(game.result(state, a), game, depth_limit - 1)
                   for a in game.actions(state))

    elif player == "CHANCE":
        total = 0
        for prob, next_state in game.chance_outcomes(state):
            total += prob * expectimax(next_state, game, depth_limit - 1)
        return total

```

## 8. Example: Stochastic Pacman
- **MAX nodes:** Pacman’s decisions (eat food, avoid ghosts).
- **CHANCE nodes:** Ghosts’ random movements.

Pacman’s goal is to **maximize expected score**, balancing:
- Immediate rewards (food, power pellets).
- Probabilistic risks (getting eaten).
- Future opportunities.

### Evaluation Function (simplified)
Eval(s) = Score(s) − 2 × DistanceToNearestGhost(s) − 10 × FoodLeft(s)
Expectimax explores possible ghost actions weighted by their movement probabilities.

## 9. Complexity
If each chance node has `d` outcomes, branching factor becomes `b × d` per level.

|Property|Expectimax|
|---|---|
|**Completeness**|Yes (finite trees)|
|**Optimality**|Yes (if probabilities known)|
|**Time Complexity**|O((b × d)^m)|
|**Space Complexity**|O(bm)|
## 10. Comparison to Other Frameworks
|Setting|Model|Example|
|---|---|---|
|**Adversarial**|Minimax|Chess|
|**Stochastic**|Expectimax|Backgammon|
|**Sequential Uncertainty**|MDP|Gridworld, Robot Planning|
|**Unknown Model**|Reinforcement Learning|Learning to play without knowing rules|

### Connections
- **MDPs (Markov Decision Processes):** Expectimax is equivalent to one-step lookahead with a known transition model.
- **Reinforcement Learning:** When transition probabilities are unknown, we must estimate them through sampling.
- **Bayesian Networks:** Expected value computations mirror probabilistic inference.

### Glossary
- **Chance Node:** Represents probabilistic outcomes with known probabilities.
- **Expected Utility:** Average of utilities weighted by probability.
- **Risk Neutrality:** Strategy that maximizes expected value, ignoring variance.
- **Transition Probability:** Probability of moving to state s′ after action a.
### takeaways
- **Expectimax** generalizes Minimax to handle randomness.
- **Chance nodes** compute weighted averages, not minima.
- Suitable for stochastic games like Backgammon or Pacman with random ghosts.
- Completeness and optimality hold when probabilities are known.
- Tends to be **risk-neutral** — it maximizes expectation, not certainty.