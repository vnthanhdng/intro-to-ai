## 1. Overview
Uninformed or blind search explores the state space without domain-specific heuristics. It only uses the problem definition (initial state, actions, transition model, goal test, and path costs). Classic strategies:
- Breadth First Search - shortest number of steps (when all step costs equal, meaning unweighted edges)
- Depth First Search - dives deep; low memory; not optimal, not complete on infinite trees.
- Uniform-Cost Search - optimal for nonnegative step costs; generalize Dijkstra
Key tradeoffs: completeness, optimally, time, space; shaped by branching factor b, solution depth d, and maximum depth m.

## 2. Problem Model
Minimal interface:
- `initial_state() -> S`
- `is_goal(s: S) -> bool`
- `successors(s: S) -> Iterable[tuple[S, action, step_cost]]`
- Optional: `hashable(s) -> Any` if states aren't naturally hashable
```python
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Tuple, List, Optional
import heapq
from collections import deque

State = Any
Action = Any

class Problem:
	def initial_state(self) -> State: ...
	def is_goal(self, s: State) -> bool: ...
	def successors(self, s: State) -> Iterable[Tuple[State, Action, float]]: ...
	def key(self, s: State) -> Any: # canonical key for visited/closed sets
		return s # override if states aren't hashable

@dataclass
class Node:
	state: State
	parent: Optional["Node"] = None
	action: Optional[Action] = None
	path_cost: float = 0.0
	depth: int = 0
	
	def path(self) -> List["Node"]:
		node, out = self, []
		while node:
			out.append(node)
			node = node.parent
		return list(reversed(out))
```


## 3. Breadth-First Search
Idea: Expand the shallowest nodes first using a FIFO queue.
Best for: Unweighted graphs where fewest actions is desired.
Properties:
- Complete? Yes, if b is finite.
- Optimal? Yes, when all step costs equal (e.g., each action cost = 1)
- Time/Space: *O(b^d)*

### Pseudocode
```scss
BFS(problem):
	start <- problem.initial_state()
	if problem.is_goal(start): return solution(start)
	
	frontier <- FIFO queue with Node(start)
	visited <- { key(start) }
	
	while frontier not empty:
		n <- frontier.pop_left()
		for (s2, a, c) in problem.successors(n.state):
			k <- key(s2)
			if k not in visited:
				child <- Node(s2, parent=n, action=a, path_cost=n.path_cost + c, depth=n.depth+1)
				if problem.is_goal(s2): return child.path()
				visited.add(k)
				frontier.push_right(child)
				
	return failure
```
### Python (for graph search)
```python
def bfs(problem: Problem) -> Optional[List[Node]]:
	start = problem.initial_state()
	if problem.is_goal(start):
		return [Node(start)]
	frontier = deque([Node(start)])
	visited = {problem.key(start)}
	while frontier:
		n = frontier.popleft()
		for s2, a, c in problem.successors(n.state):
			k = problem.key(s2)
			if k in visited:
				continue
			child = Node(s2, parent=n, action=a, path_cost=n.path_cost + c, depth=n.depth + 1)
			if problem.is_goal(s2):
				return child.path()
			visited.add(k)
			frontier.append(child)
	return None
```


## 4. Depth-First Search
Idea: Expand the deepest node first using a LIFO stack or recursion
Best for: Very large spaces where memory is tight and any solution is acceptable
Caveats: Can get stuck in deep/loops without cycle checking; not optimal.
Properties:
- Complete? Tree-DFS: No (can go infinite); Graph-DFS with cycle checking: Yes for finite graphs.
- Optimal? No.
- Time: *O(b^m)*. Space: *O(bm)* (graph) / *O(m)* (tree)

### Pseudocode (for iterative graph dfs)
```scss
DFS(problem):
	start <- problem.inital_state()
	if problem.is_goal(start): return solution(start)
	
	frontier <- stack with Node(start)
	visited <- { key(start) }
	
	while frontier not empty:
		n <- frontier.pop()
		if problem.is_goal(n.state): return n.path()
		for (s2, a, c) in reversed(problem.successors(n.state)):
			k <- key(s2)
			if k not in visited:
				visited.add(k)
				frontier.push(Node(s2, parent=n, action=a, path_cost=n.path_cost + c, depth=n.depth+1))
	
	return failure
```

### Python (iterative graph)
```python
def dfs(problem: Problem) -> Optional[List[Node]]:
	start = problem.initial_state()
	if problem.is_goal(start):
		return [Node(start)]
	stack = [Node(start)]
	visited = {problem.key(start)}
	while stack:
		n = stack.pop()
		if problem.is_goal(n.state):
			return n.path()
		# reverse expansion is optional - it just affects tie-breaking order
		succs = list(problem.successors(n.state))
		for s2, a, c in reversed(succs):
			k = problem.key(s2)
			if k not in visited:
				visited.add(k)
				stack.append(Node(s2, parent=n, action=a, path_cost=n.path_cost + c, depth=n.depth))
	
	return None
```
*For iterative deepening DFS, loop depth limits 0, 1, 2, ...: completeness of BFS with memory of DFS (Useful when d unknown)*

## 5. Uniform-Cost Search (UCS)
Idea: Expand the node with lowest path cost g(n); equivalent to Dijkstra's on the state graph.
Best for: Variable nonnegative step costs; returns least-cost path.
Properties:
- Complete? Yes, if step costs >= ε > 0 and *b* finite.
- Optimal? Yes (nonnegative costs).
- Time/Space: Up to O(b^(1 + C/ε)) in the worst case; commonly exponential in *d*. From slides: it generalizes BFS to weighted edges and must dequeue the goal to guarantee optimality.

### Pseudocode (Graph Search, with Decrease-Key via reinsert)
```scss
UCS(problem):
	start <- problem.inital_state()
	frontier <- priority queue ordered by g
	push (Node(start), g=0)
	best_g[key(start)] <- 0
	
	while frontier not empty:
		(n, g) <- pop_min(frontier)
		if g > best_g[key(n.state)]: continue // stale entry
		if problem.is_goal(n.state): return n.path()
		for (s2, a, c) in problem.successors(n.state)"
			g2 <- g + c
			k2 <- key(s2)
			if k2 not in best_g or g2 < best_g[k2]:
				best_g[k2] <- g2
				push (Node(s2, parent=n, action=a, path_cost=g2, depth=n.depth+1), g2)
	
	return failure
```

### Python (Graph Search)
```python
def ucs(problem: Problem) -> Optional[List[Node]]:
    start = problem.initial_state()
    start_key = problem.key(start)
    frontier = []
    heapq.heappush(frontier, (0.0, 0, Node(start)))  # (g, tie, node)
    best_g = {start_key: 0.0}
    tie = 1

    while frontier:
        g, _, n = heapq.heappop(frontier)
        k = problem.key(n.state)
        if g > best_g.get(k, float("inf")):
            continue  # stale
        if problem.is_goal(n.state):
            return n.path()
        for s2, a, c in problem.successors(n.state):
            g2 = g + c
            k2 = problem.key(s2)
            if g2 < best_g.get(k2, float("inf")):
                best_g[k2] = g2
                child = Node(s2, parent=n, action=a, path_cost=g2, depth=n.depth + 1)
                heapq.heappush(frontier, (g2, tie, child))
                tie += 1
    return None

```
**Why dequeue goal matters:** Enqueuing a goal early doesn't guarantee minimal cost; a cheaper path may appear later. Only when the goal is popped (lowest g) is optimality ensured.

## 6. Choosing Among BFS / DFS / UCS
| Criterion       | BFS              | DFS                                                       | UCS                           |
| --------------- | ---------------- | --------------------------------------------------------- | ----------------------------- |
| Cost Assumption | Uniform steps    | Any                                                       | Any nonnegative               |
| Completeness    | Yes (finite (b)) | Not on infinite trees (finite graphs w/ cycle-check: Yes) | Yes (ε-costs)                 |
| Optimality      | Yes (unit costs) | No                                                        | Yes                           |
| Time            | (O(b^d))         | (O(b^m))                                                  | Exponential; depends on costs |
| Space           | (O(b^d))         | (O(bm))                                                   | Exponential                   |

## 7. Subtleties & Pitfalls
- **Graph vs. Tree Search**: Failing to record **visited/closed** states can cause exponential blowup re-expanding the same states.
- **Cycle Checking**: For DFS, at least keep an **on-path** set to avoid immediate cycles; for graph search, a **global closed** set (or `best_g`) is typical.
- **Non-unit costs**: BFS is not optimal; use UCS.
- **Negative costs**: UCS is unsafe (like Dijkstra); needs nonnegative steps.

## 8. Connections to Other Things
- **Informed Search (A*)**: UCS becomes A* when we add a heuristic h(n) to order by g+h.
- **CSPs**: Uninformed backtracking ≈ DFS with constraint checks; ordering/propagation emulate “informedness.”
- **MDPs**: Expectimax/value-iteration look like search trees; UCS relates to shortest-path subroutines in policy evaluation.
### Glossary
- **Branching factor b**: max successors per node.
- **Depth d**: depth of the shallowest goal.
- **Maximum depth m**: longest path before cutoff/termination.
- **Frontier**: data structure holding nodes to expand next (queue/stack/heap).
- **Closed set**: states already expanded (or best-known g in UCS).