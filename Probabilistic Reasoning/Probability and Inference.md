## 1. Motivation

Up to now, we’ve assumed **perfect knowledge** of the environment — deterministic or stochastic but known.  
However, in many AI systems, **uncertainty** is unavoidable:
- Sensors are noisy (e.g., robots, self-driving cars)
- Outcomes are uncertain (e.g., medical diagnoses, weather)
- States are hidden (e.g., speech recognition, tracking)

We need **probability theory** to represent and reason about uncertain knowledge.

---

## 2. Random Variables and Domains

A **random variable (RV)** represents an uncertain quantity that can take on multiple possible values.

Examples:
- Weather ∈ {Sunny, Cloudy, Rainy}
- DiceRoll ∈ {1, 2, 3, 4, 5, 6}
- SensorReading ∈ ℝ

A **probability distribution** P(X) specifies the likelihood of each possible value.

Example:  
P(Weather) = {Sunny: 0.7, Cloudy: 0.2, Rainy: 0.1}
Sum over all possible outcomes = 1.  
$$
\sum_x P(X = x) = 1  
$$

---

## 3. Joint and Marginal Distributions

- **Joint distribution** P(X, Y): probability over combinations of values of X and Y.  
    Example: P(Weather, Traffic) gives probabilities of both conditions.
- **Marginalization:** computing probabilities of subsets by summing over the rest.  
    Example:  
    $$ P(X) = \sum_y P(X, Y = y)  $$

This allows us to “forget” irrelevant variables.

---

## 4. Conditional Probability

Conditional probability represents belief in X given evidence Y.
$$
P(X|Y) = \frac{P(X, Y)}{P(Y)}  
$$

Interpretation: “Given Y is true, how likely is X?”

Example:  
P(Rain | Clouds) = P(Rain ∧ Clouds) / P(Clouds)

---

## 5. Independence and Conditional Independence

### Independence

X and Y are independent if:  
$$
P(X, Y) = P(X) P(Y)  
$$
Equivalently:  
$$ 
P(X|Y) = P(X)  
$$

Example: Rolling two dice — outcomes don’t influence each other.

### Conditional Independence

X and Y are conditionally independent given Z if:  
$$ 
P(X, Y | Z) = P(X | Z) P(Y | Z)  
$$  
Example:  
If you know it’s raining (Z), then whether your shoes are wet (X) is independent of whether the grass is wet (Y).

Conditional independence is the backbone of efficient probabilistic reasoning (used in Bayesian networks).

---

## 6. Product Rule and Chain Rule

**Product Rule:**  
$$  
P(X, Y) = P(X | Y) P(Y)  
$$

**Chain Rule:**  
For variables X₁, X₂, …, Xₙ,  
$$ 
P(X₁, X₂, ..., Xₙ) = P(X₁) P(X₂ | X₁) P(X₃ | X₁, X₂) ... P(Xₙ | X₁, X₂, ..., Xₙ₋₁)  
$$

This decomposes large joint distributions into conditional probabilities.

---

## 7. Bayes’ Rule

Bayes’ Rule lets us invert conditional probabilities — a cornerstone of probabilistic inference.
$$  
P(X|Y) = \frac{P(Y|X) P(X)}{P(Y)}  
$$

Where:

- P(X): prior — initial belief about X.
- P(Y|X): likelihood — how evidence Y behaves given X.
- P(X|Y): posterior — updated belief about X after seeing Y.
- P(Y): normalization constant ensuring probabilities sum to 1.

Example:  
If you test positive for a disease,  
$$  
P(Disease|Positive) = \frac{P(Positive|Disease) P(Disease)}{P(Positive)}  
$$  
depends on both accuracy of the test and disease base rate.

---

## 8. Example: Medical Diagnosis

Suppose:

- P(Disease) = 0.01
- P(Positive|Disease) = 0.9
- P(Positive|¬Disease) = 0.05

Compute P(Disease|Positive):

$$  
P(Disease|Positive) = \frac{0.9 × 0.01}{0.9 × 0.01 + 0.05 × 0.99} = 0.154  
$$

Even with a 90% accurate test, the true probability of being sick given a positive result is only 15%!  
This is due to the **base rate fallacy**.

---

## 9. Normalization

Sometimes we can compute only proportional probabilities.  
Normalize to get actual probabilities:

$$ 
P(X|Y) = \alpha P(X, Y)  
$$  
where α = 1 / ∑ₓ P(X, Y)

This trick is widely used in Bayesian networks and filtering.

---

## 10. Inference by Enumeration

Given a full joint distribution, we can answer any query by summing out irrelevant variables.

Example: compute P(Burglary | Alarm = true)

Algorithm:
1. For each possible value of B:
    - sum over all other variables (E, A, J, M, …)
    - multiply their conditional probabilities
2. Normalize.
    

Though exact, this method scales **exponentially** with the number of variables.

---

## 11. The Ghostbusters Example (CS188)

Imagine Pacman trying to locate a hidden ghost.
- Sensors provide noisy distance measurements.
- Each cell in the grid has prior probability of containing the ghost.
- After observing new evidence, Pacman updates beliefs using Bayes’ Rule.

Belief update:  
$$  
P(Ghost|Sensor) ∝ P(Sensor|Ghost) P(Ghost)  
$$

This forms the foundation of **tracking**, **filtering**, and **belief updating** in uncertain environments.

---

## 12. Probability vs Logic

|Logic|Probability|
|---|---|
|Truth values|Degrees of belief (0–1)|
|Inference: deterministic|Inference: uncertain|
|Knowledge base: facts|Knowledge base: distributions|
|“If A then B”|P(B|
|Rigid|Flexible, robust to noise|

Probability extends logic by allowing reasoning under incomplete or noisy data.

---

## 13. Applications

- **Medical diagnosis:** reasoning from symptoms.
- **Speech recognition:** decoding words from noisy audio.    
- **Spam filtering:** classifying messages by word probabilities.
- **Robot localization:** estimating position from uncertain sensors.
- **Autonomous driving:** predicting pedestrian movements.

### Connections

- **From RL to Probabilistic Reasoning:**  
    RL uses expectations over rewards; inference uses expectations over hidden variables.
- **To Bayesian Networks:**  
    Bayes’ Rule and conditional independence form their foundation.
- **To Machine Learning:**  
    Probabilistic models underlie Naive Bayes, Hidden Markov Models, and neural network likelihoods.

---

### Glossary

- **Random Variable (RV):** A variable with uncertain value.    
- **Joint Distribution:** Probability over multiple variables.
- **Marginal Probability:** Probability over one variable regardless of others.
- **Conditional Probability:** Probability of one variable given another.
- **Independence:** When one variable provides no information about another.
- **Bayes’ Rule:** Method for inverting conditionals to update beliefs.
### Key Takeaways

- Probability theory provides a framework for reasoning under uncertainty.
- Conditional and joint distributions allow flexible modeling of dependencies.
- Bayes’ Rule enables inference: updating beliefs with evidence.
- Independence and conditional independence reduce computation drastically.
- Inference by enumeration is exact but computationally expensive — motivates Bayesian networks and sampling methods.