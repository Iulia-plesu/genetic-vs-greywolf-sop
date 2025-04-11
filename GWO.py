import random

n = 4
cost = [
    [0, 5, 0, 9],
    [0, 0, 7, 0],
    [0, 0, 0, 2],
    [0, 0, 0, 0]
]

precedence = [(0, 2), (1, 3)]

# -------------------------------
# SOP Functions
# -------------------------------
def satisfies_constraints(chromosome):
    """Check if the chromosome respects the precedence constraints."""
    pos = {gene: idx for idx, gene in enumerate(chromosome)}
    for a, b in precedence:
        if pos[a] > pos[b]:
            return False
    return True


def total_cost(chromosome):
    """Calculate the total cost of the solution.

    If any connection between consecutive nodes is invalid (infinity), returns infinity.
    """
    total = 0
    for i in range(len(chromosome) - 1):
        if cost[chromosome[i]][chromosome[i + 1]] == float("inf"):
            return float("inf")
        total += cost[chromosome[i]][chromosome[i + 1]]
    return total if total != 0 else float("inf")


def fitness(chromosome):
    """Fitness is defined as the inverse of total cost (plus a small constant for numerical stability)."""
    return 1 / (total_cost(chromosome) + 1e-6)


def generate_valid_solution(max_attempts=1000):
    """Generate a valid solution by shuffling nodes until the constraints are met."""
    attempts = 0
    while attempts < max_attempts:
        p = list(range(n))
        random.shuffle(p)
        if satisfies_constraints(p):
            return p
        else:
            attempts += 1

    print("Maximum attempts reached to generate a valid solution.")
    return None


def combine(alpha, beta, delta):
    """Combine three solutions (alpha, beta, and delta) to create a new solution."""
    new_solution = []
    for i in range(n):
        options = [alpha[i], beta[i], delta[i]]
        for o in options:
            if o not in new_solution:
                new_solution.append(o)
                break

    missing = [x for x in range(n) if x not in new_solution]
    random.shuffle(missing)
    new_solution += missing

    if satisfies_constraints(new_solution):
        return new_solution

    return generate_valid_solution()


