## 1. Overview
Uninformed or blind search explores the state space without domain-specific heuristics. It only uses the problem definition (initial state, actions, transition model, goal test, and path costs). Classic strategies:
- Breadth First Search - shortest number of steps (when all step costs equal, meaning unweighted edges)
- Depth First Search - dives deep; low memory; not optimal, not complete on infinite trees.
- Uniform-Cost Search - optimal for nonnegative step costs; generalize Dijkstra

