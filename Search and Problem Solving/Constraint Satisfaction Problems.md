
## 1. Overview

A **Constraint Satisfaction Problem (CSP)** is defined by:

- **Variables:** $$  X_1, X_2, \dots, X_n $$
- **Domains:** $$ D_1, D_2, \dots, D_n $$
- **Constraints:** Rules restricting combinations of variable assignments.

Each constraint limits the values that variables can take **together**.  
A solution is a **complete assignment** satisfying **all constraints**.

Examples:
- **Map Coloring:** Adjacent regions must have different colors.  
- **Sudoku:** Each row, column, and 3×3 grid contains digits 1–9 exactly once.  
- **N-Queens:** No two queens attack each other.

---

## 2. CSP vs. Standard Search

| Aspect | Standard Search | CSP |
|---------|------------------|-----|
| **State** | Arbitrary structure | Variable assignments |
| **Goal Test** | Black-box predicate | Constraints |
| **Successor Function** | Generates next states | Assign next variable |
| **Solution** | Path to goal | Complete consistent assignment |

Because CSPs have structure, we can exploit it to prune search and reason more efficiently.

---

## 3. Backtracking Search

**Idea:**  
Perform **depth-first search** over partial assignments, but check constraints at each step to **fail early**.

### Pseudocode
```scss
BACKTRACK(assignment):  
	if assignment is complete: 
		return assignment
	var ← SELECT-UNASSIGNED-VARIABLE(assignment)

	for value in ORDER-DOMAIN-VALUES(var, assignment):
	    if value is consistent with assignment:
	        add {var = value} to assignment
	        result ← BACKTRACK(assignment)
	        if result ≠ failure:
	            return result
	        remove {var = value} from assignment

	return failure

```
### Key points
- Basic depth-first framework.
- Calls to helper functions let us add **heuristics** for dynamic ordering (e.g., MRV, LCV) and **filtering** (e.g., forward checking).
- 4 improvements:
	- General purpose ideas can give huge gains in speed but it's all still NP-hard
	- Ordering
		- Which variable should be assigned next? MRV
		- In what order should its values be tried? LCV
	- Filtering: Can we detect inevitable failure early?
	- Structure: Can we exploit the problem structure?

---

## 4. Variable and Value Heuristics

### Minimum Remaining Values (MRV) or Most Constrained Variables (MCV)
- Choose the variable with the **fewest legal values** left.
- Encourages “fail-fast”: try the most constrained variable first.

### Degree Heuristic
- Tie-breaker for MRV: choose the variable involved in the **most constraints** on unassigned variables.

### Least Constraining Value (LCV)
- When ordering values, try those that **rule out the fewest options** for neighboring variables.

---

## 5. Constraint Propagation

**Filtering** methods prune variable domains by enforcing consistency between variables before deeper search.

### Arc Consistency
An arc $$ X \to Y $$ is **consistent** if:
> for every value `x` in `X`, there exists some value `y` in `Y` satisfying the constraint between them.

If no such `y` exists, remove `x` from `X`’s domain.

Strong K-consistency: also k-1, k-2,..., 1 consistent. 
Claim: strong n-consistency means we can solve without backtracking.

---

### AC-3 Algorithm (Arc Consistency 3)

**Goal:** Enforce arc consistency across all constraints.
Mnemonic: Always delete from the tail.

### Pseudocode
```scss
function AC-3(csp): 
	queue ← a queue of all arcs (Xi, Xj) in csp 
	while queue is not empty: 
		(Xi, Xj) ← queue.pop()
		
		 if REVISE(csp, Xi, Xj): 
			if domain[Xi] is empty: 
				 return false // Inconsistency found
			for each Xk in neighbors[Xi] - {Xj}: 
				add (Xk, Xi) to queue 
				
	return true // Arc consistent

```

```scss
function REVISE(csp, Xi, Xj): 
	revised ← false 
	
	for each x in domain[Xi]: 
		if no value y in domain[Xj] satisfies the constraint between Xi and Xj: 
			remove x from domain[Xi] 
			revised ← true 
	return revised
```

**Runtime:** $$O(n^2 d^3)$$ can be reduced to $$O(n^2 d^2) $$

## 6. Python Example — AC-3

```python
from collections import deque

def ac3(csp):
    queue = deque([(Xi, Xj) for Xi in csp.variables for Xj in csp.neighbors[Xi]])
    while queue:
        Xi, Xj = queue.popleft()
        if revise(csp, Xi, Xj):
            if not csp.domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True

def revise(csp, Xi, Xj):
    revised = False
    for x in set(csp.domains[Xi]):
        if not any(csp.constraints(Xi, x, Xj, y) for y in csp.domains[Xj]):
            csp.domains[Xi].remove(x)
            revised = True
    return revised```

## 7. Forward Checking (one-step lookahead)

After assigning a variable, eliminate inconsistent values from neighboring domains.

### Pseudocode

```scss
FORWARD-CHECKING(csp, var, value):
    for each neighbor in neighbors[var]:
        for each val in domain[neighbor]:
            if constraint(var, value, neighbor, val) is violated:
                remove val from domain[neighbor]
                if domain[neighbor] becomes empty:
                    return false
    return true

```

Combines well with **MRV** — quickly detects dead-ends before deep recursion.

## 8. Example: Map Coloring

### Variables

`WA, NT, SA, Q, NSW, V, T`
### Domains
`{red, green, blue}` for each state

### Constraints
Adjacent states cannot have the same color.
Backtracking + MRV + AC-3 finds a solution like:
`WA = red NT = green SA = blue Q  = red NSW = green V  = red T  = blue`

---
## 9. Structure and Decomposition

- **Independent Subproblems:** Separate constraint graphs can be solved independently.
- **Tree-Structured CSPs:** Solvable in $$ O(nd^2) (linear time).$$
	Algorithm for tree-structured CSPs:
	- Order: choose a root variable, order variables so that parents precede children.
	- Remove backward: For i = n : 2, apply RemoveInconsistent(Parent(X_i), X_j) (arc consistency from sub tree and building up to the root)
	- Assign forward: For i = 1 : n, assign X consistently with Parent(X_i)
- **Cutset Conditioning:** Assign a small subset of variables to break cycles, leaving a tree.
	Nearly Tree-Structured CSPs
	- Conditioning: instantiate a variable, prune its neighbors' domains
	- Cutset conditioning: instantiate (in all ways) a set of variables such that the remaining constraint graph is a tree
	- Cutset size c gives runtime $$ O((d^c) (n -c) d^2) $$, which is very fast for small c.


---

## 10. Iterative Improvement for CSPs

When domains are large or random restarts help (e.g., **N-Queens**):
### Min-Conflicts Algorithm

1. Start with a complete assignment.
2. While not solved:
    - Pick a conflicted variable.
    - Assign a value that minimizes number of constraint violations.
Efficient for large random CSPs.

### Connections

- **Search Algorithms:** Backtracking = DFS; CSP-specific pruning adds “informedness.”
- **Probabilistic Reasoning:** Arc consistency ≈ variable elimination (both prune inconsistent states).
- **Learning:** Reinforcement learning agents often rely on constraint optimization for planning.

### Glossary
|Term|Definition|
|---|---|
|**Variable**|Symbol whose value is to be determined|
|**Domain**|Possible values for a variable|
|**Constraint**|Rule restricting allowable value combinations|
|**Consistency**|All constraints are satisfied|
|**Arc Consistency**|Each value of one variable has support in its neighbor|
|**Backtracking Search**|DFS that incrementally builds and checks assignments|

### some takeaways
- CSPs formalize problems as **variables + domains + constraints**.
- **Backtracking** with **MRV**, **LCV**, and **forward checking** drastically reduces search.
- **Arc Consistency (AC-3)** propagates constraints efficiently.
- Exploiting **problem structure** (trees, subgraphs) leads to huge speedups.