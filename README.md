# 🧠 Sequential Ordering Problem (SOP) Solver

This project provides two powerful metaheuristic approaches to solve the **Sequential Ordering Problem (SOP)**:

- 🐺 **Grey Wolf Optimizer (GWO)**
- 🧬 **Genetic Algorithm (GA)**

The Sequential Ordering Problem is an extension of the Asymmetric Traveling Salesman Problem (ATSP), where not only is the shortest path through all nodes required, but **precedence constraints** must also be satisfied. These constraints enforce that certain nodes must appear before others in the final route.

---

## 📌 Problem Definition

Given:

- A set of nodes (tasks or cities)
- A cost matrix defining the asymmetric travel cost between nodes
- A list of precedence constraints (e.g., "Node A must come before Node B")

**Objective:**  
Find the optimal permutation of nodes such that:

- Each node is visited exactly once
- All precedence constraints are satisfied
- The total path cost is minimized

---

## 🚀 Features

### ✅ General Features

- Reads `.sop` files in TSPLIB/SOPLIB format
- Handles missing precedence constraints
- Converts `-1` values to infinity (i.e., invalid transitions)
- Automatically builds the cost matrix
- Ensures all solutions are valid and satisfy constraints

---

### 🐺 Grey Wolf Optimizer (GWO)

- Population-based heuristic inspired by grey wolf hunting behavior
- Alpha, Beta, Delta wolves guide solution construction
- Uses recombination and mutation for generating new solutions
- Efficient solution repair to maintain constraint validity
- Fast convergence for small/medium SOP instances
- Randomized mutations with constraint checks
- Logs progress and final cost at each generation

---

### 🧬 Genetic Algorithm (GA)

- Initial population generated with only valid solutions
- Tournament selection to choose parents
- Order crossover (OX) operator used for child creation
- Mutation by swapping two nodes
- Precedence-aware repair ensures all offspring are valid
- Configurable population size and mutation rate
- Tracks best solution over multiple generations

---

### 📦 Requirements

```bash
Python 3.x
