## 1. Environment Types
An environment defines the world in which an AI agent operates. Classifying environments helps us understand what algorithms and agent designs are appropriate.

**Key Dimensions**
- Fully vs. Partially Observable
	- Fully observable: Agent has access to the complete state of the environment
		- Example: Chess (full board is visible to both players)
	- Partially observable: Agent only perceives limited information.
		- Example: Poker (hidden cards)
- Single Agent vs. Multi-Agent
	- Single Agent: Agent acts alone.
		- Solving a crossword puzzles.
	- Multi-agent: Multiple entities act simultaneously.
		- Cooperative (shared goals): Autonomous cars coordinating traffic.
		- Competitive (conflicting goals): Pacman vs. Ghosts
		- Communication may be explicit (messaging) or implicit (inferring from actions).
- Discrete vs. Continuous
	- Discrete: Limited set of states/actions.
		- Example: Tic-Tac_Toe (finite spaces, moves).
	- Continuous: Infinite state/action space.
		- Example: Driving a car (continuous positions, speeds)
- Deterministic vs. Stochastic
	- Deterministic: Next state is fully determined by current state and action.
		- Example: Sliding puzzle.
	- Stochastic: Outcomes involve randomness or uncertainty.
		- Example: Dice rolls in board games.
- Episodic vs. Sequential
	- Episodic: Experience broken into independent episodes.
		- Example: Image classification (each image is separate).
	- Sequential: Current decisions affect future states.
		- Example: Chess, self-driving
- Known vs. Unknown
	- Known: Agent fully understands rules and dynamics.
	- Unknown: Agent must learn or infer them through interaction.

## 2. Types of Agents
Agents differ by how they use information and make decisions.
- Simple Reflex Agents
	- Act only on current percept (no memory)
	- Implemented via condition-action rules (`if condition then action`)
	- Example: Thermostat.
	- Limitation: Can't handle partial observability or long-term planning.
- Model-Based Reflex Agents
	- Maintain internal state (memory) of the environment.
	- Use model of world dynamics to interpret current percepts
	- Example: Self-driving car considering unseen cars at intersections.
- Goal-Based Agents
	- Choose actions by considering future states and goals.
	- Involves planning: searching through possible action sequences.
	- Example: Chess program aiming to checkmate.
- Utility-Based Agents
	- Extend goal-based agents with utility function: measure of desirability
	- Choose actions to maximize expected utility
	- Example: Navigation system that balances speed vs. safety
- Learning Agents
	- Improve performance over time by learning from feedback.
	- Components:
		1. Performance Element - decides actions.
		2. Learning Element - improve the agent.
		3. Critic - gives feedback.
		4. Problem Generator - suggests exploratory actions.
	- Example: Reinforcement learning agents in robotics.

### Key Takeaways
- AI environments can be classified along multiple dimensions (observability, determinism, etc.)
- Different agent architectures (reflex -> learning) correspond to increasing levels of sophistication.
- Designing an intelligent agent requires matching the agent type to environment properties.

