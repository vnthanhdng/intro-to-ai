## 1. Overview
Informed (heuristic) search augments path cost with problem-specific knowledge via a heuristic function h(n) that estimates the cheapest cost from node n to a goal. Properly designed heuristics can make search dramatically more efficient and still guarantee optimality (with the right conditions).
Core algorithms:
- Greedy Best-First Search (GBFS) -- expand the node with the smallest h(n). Fast but not optimal.
- A* -- expand the node with the smallest f(n) = g(n) + h(n). Optimal with admissible (tree search) / consistent (graph search) heuristics.

## 2. Heuristics
A heuristic h(x) estimates distance-to-goal. Examples: Manhattan distance, Euclidean distance (pathing), tile puzzles.

### Admissibility (optimistic)
A heuristic is **admissible** if it **never overestimates** the true least cost `h*(n)` to a goal:

$$

0 \le h(n) \le h^*(n) \text{ for all } n


$$
Guarantees **optimality** of A\* in **tree search**.

### Consistency (Monotonicity)
A heuristic is **consistent** if, for every edge `n → n'` with cost `c(n, n')`:
$$

h(n) \le c(n, n') + h(n'), \quad h(\text{goal}) = 0

$$

Guarantees **optimality** of A\* in **graph search** and ensures that `f(n)` never decreases along a path.

Questions
1. If we used an inadmissible heuristic in A* graph search, would the search be complete? Would it be optimal?
	If a heuristic function is bounded, then A* graph search would visit all the nodes eventually, and would find a path to the goal state if there exists one. An inadmissible heuristic does not guarantee optimality as it can make the good optimal goal look as though it is very far off, and take you to a suboptimal goal.
2. If we used an admissible heuristic in A* graph search, is it guaranteed to return an optimal solution? What if the heuristic was consistent?
	Admissible heuristics do not necessarily guarantee optimality; they are only guaranteed to return an optimal solution for graph search if they are consistent as well.
3. A general advantage that an inadmissible heuristic might have over an admissible one?
	The time to solve an A* search problem is a function of two factors: the number of nodes expanded, and the time spent per node. An inadmissible heuristic may be faster to computer, leading to a solution that is obtained faster due to less time spent per node. It can also be a closer estimate to the actual cost function (even through at times it will overestimate), thus expanding less nodes. We lose the guarantee of optimality by using an inadmissible heuristic. But sometimes we may be okay with finding a suboptimal solution to a search problem.
### Designing Heuristics (Relaxed Problems)
Powerful admissible heuristics often come from **relaxing constraints** so the problem becomes easier.  
The relaxed problem’s solution cost is a valid lower bound.

**8-Puzzle examples**
- *Misplaced Tiles*: number of tiles out of place — admissible  
- *Manhattan Distance*: sum of |Δrow| + |Δcol| for all tiles — stronger admissible  
- *Dominance*: if `h₁ ≥ h₂` for all n, then `h₁` dominates `h₂` (fewer expansions)

---

## 3. Greedy Best-First Search (GBFS)

**Idea:** prioritize nodes that *seem* closest to the goal by `h(n)` (ignore `g`).  
Fast and often effective, but **not optimal** and may **thrash**.

### Pseudocode (Graph Search)
```
GREEDY-BEST-FIRST-SEARCH(problem, h):  
	start ← problem.initial_state()  
	if problem.is_goal(start):  
		return solution(start)
	frontier ← priority queue ordered by h(s)
	push Node(start) with key h(start)
	closed ← ∅

	while frontier not empty:
	    n ← pop_min(frontier)
	    if problem.is_goal(n.state):
	        return n.path()

	    k ← key(n.state)
	    if k in closed:
	        continue
	    closed.add(k)

	    for (s2, a, c) in problem.successors(n.state):
	        if key(s2) not in closed:
	            child ← Node(s2, parent=n, action=a,
                         path_cost=n.path_cost + c,
                         depth=n.depth + 1)
	            push child with key h(s2)

	return failure
```

### Python
```python
import heapq

def greedy_best_first_search(problem, h):
    start = problem.initial_state()
    if problem.is_goal(start):
        return [Node(start)]
    frontier = []
    tie = 0
    heapq.heappush(frontier, (h(start), tie, Node(start)))
    closed = set()

    while frontier:
        _, _, n = heapq.heappop(frontier)
        if problem.is_goal(n.state):
            return n.path()
        k = problem.key(n.state)
        if k in closed:
            continue
        closed.add(k)
        for s2, a, c in problem.successors(n.state):
            k2 = problem.key(s2)
            if k2 not in closed:
                tie += 1
                heapq.heappush(frontier, (h(s2), tie,
                               Node(s2, parent=n, action=a,
                                    path_cost=n.path_cost + c, depth=n.depth + 1)))
    return None
```

## 4. A* Search
**Key idea:** order expansions by `f(n) = g(n) + h(n)`.

- **Tree Search:** admissible `h` ⇒ optimal
- **Graph Search:** consistent `h` ⇒ optimal
- **Stop rule:** do **not** stop when you enqueue a goal; stop when you **dequeue** it (first time popped).

### Pseudocode (Graph Search)
```scss
A-STAR(problem, h):
  start ← problem.initial_state()
  g_best[key(start)] ← 0
  frontier ← priority queue by f = g + h
  push Node(start) with f = 0 + h(start)

  while frontier not empty:
    n ← pop_min(frontier)
    if problem.is_goal(n.state): return n.path()
    for (s2, a, c) in problem.successors(n.state):
      g2 ← n.path_cost + c
      k2 ← key(s2)
      if g2 < g_best.get(k2, +∞):
        g_best[k2] ← g2
        f2 ← g2 + h(s2)
        push Node(s2, parent=n, action=a, path_cost=g2, depth=n.depth+1) with key f2
  return failure

```

### Python
```python
def astar(problem, h):
    start = problem.initial_state()
    start_key = problem.key(start)
    g_best = {start_key: 0.0}
    frontier = []
    tie = 0
    heapq.heappush(frontier, (h(start), tie, Node(start)))

    while frontier:
        f, _, n = heapq.heappop(frontier)
        if problem.is_goal(n.state):
            return n.path()
        for s2, a, c in problem.successors(n.state):
            g2 = n.path_cost + c
            k2 = problem.key(s2)
            if g2 < g_best.get(k2, float("inf")):
                g_best[k2] = g2
                tie += 1
                heapq.heappush(frontier, (g2 + h(s2), tie,
                               Node(s2, parent=n, action=a, path_cost=g2, depth=n.depth + 1)))
    return None

```

## 5. Properties & Proof Sketches
- **A* Tree Search Optimality (admissible h):**  
    The optimal goal `A` leaves the fringe before any suboptimal goal `B`, because some ancestor `n` of `A` has  
    `f(n) ≤ f(A) < f(B)`.
- **A* Graph Search Optimality (consistent h):**  
    `f` never decreases along paths; nodes reaching a state with the lowest `g` are expanded before suboptimal alternatives; the first time goal is popped, it’s optimal.
## 6. Greedy vs UCS vs A*
| Algorithm           | Priority Key  | Optimal?                             | Notes                                      |
| ------------------- | ------------- | ------------------------------------ | ------------------------------------------ |
| Greedy Best-First   | `h(n)`        | No                                   | Fast but can “bee-line” to the wrong goal  |
| Uniform-Cost Search | `g(n)`        | Yes                                  | Heuristic-free; explores by path cost      |
| A*                  | `g(n) + h(n)` | Yes (with admissible/consistent `h`) | Balances known cost and estimated distance |
## 7. Heuristic Quality & Work
- **Stronger heuristics** (larger but still admissible) ⇒ fewer node expansions.
- There’s a **trade-off**: better `h` may be more expensive to compute per node.

## 8. Connections
- **UCS ↔ A***: UCS is A* with `h ≡ 0`.
- **CSPs:** heuristic variable/value ordering parallels A* guidance.
- **Game Search:** evaluation functions & node ordering in alpha-beta resemble heuristic design.

### Glossary
| Term           | Definition                             |
| -------------- | -------------------------------------- |
| **g(n)**       | Cost from start to `n`                 |
| **h(n)**       | Estimated cost from `n` to goal        |
| **f(n)**       | Total estimated cost (`g + h`)         |
| **Admissible** | Never overestimates true cost          |
| **Consistent** | Obeys triangle inequality across edges |
### Takeaways
- **Greedy** is fast but not guaranteed optimal.
- A* is optimal when using admissible/consistent heuristics.    
- Stronger admissible heuristics reduce node expansions exponentially.
- Relaxed problems are the foundation for heuristic design.