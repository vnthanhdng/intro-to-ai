## 1. Overview 
In adversarial search, the environment contains multiple agents with conflicting goals. Typical example: two-player games such as Chess, Tic-Tac-Toe, or Pacman vs. Ghosts. 

Each agent chooses actions alternately, and one's gain is the other's loss -- a zero-sum setting.

Game properties:
- Deterministic games: Chess, Checkers
- Stochastic games: Backgammon (involves dice)
- Perfect information: Tic-Tac-Toe
- Imperfect information: Poker, Battleship

Goal: compute a strategy (policy) that maximizes the agent's expected utility, assuming the opponent plays optimally.

## 2. Game Tree
- Nodes represent game states.
- Edges represent actions or moves.
- Levels alternate between MAX and MIN players.
- Terminal nodes represent leaf states with known utilities U(s).

## 3. The Minimax Algorithm
Idea: choose actions assuming the opponent plays optimally to minimize your utility. You aim to maximize the minimum possible outcome -- hence "minimax" or "maximin"

Mathematically:
Minimax(s) = 
- If s is terminal: U(s)
- If MAX's turn: max over actions of min over successors Minimax (s')
- If MIN's turn: min over actions of max over successors Minimax (s')

## 4. Minimax Pseudocode
```scss
MINIMAX-DECISION(state):  
 value, best_action ← MAX-VALUE(state)  
 return best_action

MAX-VALUE(state):  
 if TERMINAL(state): return UTILITY(state), None  
 v ← −∞  
 best_action ← None  
 for action in ACTIONS(state):  
  value, _ ← MIN-VALUE(RESULT(state, action))  
  if value > v:  
   v ← value  
   best_action ← action  
 return v, best_action

MIN-VALUE(state):  
 if TERMINAL(state): return UTILITY(state), None  
 v ← +∞  
 best_action ← None  
 for action in ACTIONS(state):  
  value, _ ← MAX-VALUE(RESULT(state, action))  
  if value < v:  
   v ← value  
   best_action ← action  
 return v, best_action
```
Properties:
- Complete if the game tree is finite
- Optimal if both players play optimally.
- Time complexity *O(b^m)*
- Space complexity *O(bm)* for DFS recursion

## 5. Python Implementation (basic)
```python
def minimax_decision(state, game):  
	"""  
	Returns the best action for MAX in the given state.  
	game must define:  
	- actions(state)  
	- result(state, action)  
	- utility(state)  
	- terminal_test(state)  
	- to_move(state): 'MAX' or 'MIN'  
	"""  
	player = game.to_move(state)
	
	def max_value(s):
	    if game.terminal_test(s):
	        return game.utility(s, player)
	    v = float("-inf")
	    for a in game.actions(s):
	        v = max(v, min_value(game.result(s, a)))
	    return v

	def min_value(s):
	    if game.terminal_test(s):
	        return game.utility(s, player)
	    v = float("inf")
	    for a in game.actions(s):
	        v = min(v, max_value(game.result(s, a)))
	    return v

	best_action = max(game.actions(state), key=lambda a: min_value(game.result(state, a)))
	return best_action

```

## 6. Evaluation Functions
In complex games, the tree cannot be searched to the end.  
Use an **evaluation function** Eval(s) to approximate utility at a cutoff depth.
Example:
- Chess: weighted sum of piece values, board control, mobility.
- Pacman: food left, distance to ghosts, score.
General form:  
Eval(s) = w₁·f₁(s) + w₂·f₂(s) + … + wₖ·fₖ(s)

## 7. Depth-Limited Minimax

When trees are too deep, limit the search depth d:

DEPTH-LIMITED-MINIMAX(state, depth):  
 if TERMINAL(state) or depth = 0:  
  return EVAL(state)  
 …

This enables real-time decision making with heuristic cutoffs.

## 8. Alpha–Beta Pruning
Motivation:  
Minimax examines the entire game tree — exponential in depth.  
We can prune branches that cannot affect the final decision.

Definitions:
- α: best (highest) value achievable by MAX so far.
- β: best (lowest) value achievable by MIN so far.
Rule: prune when α ≥ β.

## 9. Alpha-Beta Pseudocode
```scss
ALPHA-BETA-SEARCH(state):  
 value, best_action ← MAX-VALUE(state, −∞, +∞)  
 return best_action

MAX-VALUE(state, α, β):  
 if TERMINAL(state): return UTILITY(state), None  
 v ← −∞  
 best_action ← None  
 for action in ACTIONS(state):  
  value, _ ← MIN-VALUE(RESULT(state, action), α, β)  
  if value > v:  
   v ← value  
   best_action ← action  
  if v ≥ β: return v, best_action // β-cutoff  
  α ← max(α, v)  
 return v, best_action

MIN-VALUE(state, α, β):  
 if TERMINAL(state): return UTILITY(state), None  
 v ← +∞  
 best_action ← None  
 for action in ACTIONS(state):  
  value, _ ← MAX-VALUE(RESULT(state, action), α, β)  
  if value < v:  
   v ← value  
   best_action ← action  
  if v ≤ α: return v, best_action // α-cutoff  
  β ← min(β, v)  
 return v, best_action
```

## 10. Python Implementation (Alpha-Beta)
```python
def alphabeta_search(state, game, depth_limit=4):  
	player = game.to_move(state)
	
	def max_value(s, alpha, beta, depth):
	    if game.terminal_test(s) or depth == 0:
	        return game.utility(s, player)
	    v = float("-inf")
	    for a in game.actions(s):
	        v = max(v, min_value(game.result(s, a), alpha, beta, depth - 1))
	        if v >= beta:
	            return v  # β-cutoff
	        alpha = max(alpha, v)
	    return v

	def min_value(s, alpha, beta, depth):
	    if game.terminal_test(s) or depth == 0:
	        return game.utility(s, player)
	    v = float("inf")
	    for a in game.actions(s):
	        v = min(v, max_value(game.result(s, a), alpha, beta, depth - 1))
	        if v <= alpha:
	            return v  # α-cutoff
	        beta = min(beta, v)
	    return v

	best_action = max(game.actions(state),
                  key=lambda a: min_value(game.result(state, a),
                                          float("-inf"), float("inf"), depth_limit))
	return best_action
```

## 11. Properties of Alpha–Beta
- Returns the same action as Minimax.
- Never affects correctness.
- Best-case time complexity: O(b^(m/2)) — effectively doubles search depth.
- Worst-case: O(b^m).
- Space complexity: O(bm).

Ordering of actions is critical: exploring the best moves first increases pruning effectiveness.

---

## 12. Real-World Example — Pacman

In multi-agent Pacman:
- Pacman = MAX player
- Ghosts = MIN players (adversaries)

Evaluation function may include:
- Remaining food
- Distance to nearest ghost
- Current game score

### Connections
- Expectimax handles stochastic outcomes (chance nodes).
- MDPs model probabilistic rather than adversarial uncertainty.
- Reinforcement learning self-play uses minimax-like reasoning for policy learning.

### Glossary
- **Minimax:** Decision rule that maximizes the minimum gain.
- **Alpha–Beta Pruning:** Optimization technique to skip branches that cannot affect the result.
- **Utility Function:** Terminal payoff of a state.
- **Evaluation Function:** Heuristic estimate of utility at cutoff depth.
- **Cutoff Depth:** Depth limit for practical search.

## takeaways
- Minimax models optimal play in zero-sum games.
- Alpha–Beta pruning reduces search drastically without changing the result.
- Evaluation functions make search feasible in large or complex games.
- Move ordering can make the difference between exponential and quadratic exploration.