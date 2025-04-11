import random
import time
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

def gwo_sop(num_wolves=20, max_iter=100):
    # Initialize population with valid solutions.
    population = []
    for _ in range(num_wolves):
        sol = generate_valid_solution()
        if sol is None:
            return None, float("inf")
        population.append(sol)

    start_time = time.time()  # Timer start

    for gen in range(max_iter):
        population.sort(key=fitness, reverse=True)
        alpha = population[0]
        beta = population[1]
        delta = population[2]

        new_wolves = [alpha, beta, delta]  # Elitism: preserving top three solutions

        while len(new_wolves) < num_wolves:
            new_sol = combine(alpha, beta, delta)
            if random.random() < 0.3:  # Small mutation probability
                i, j = random.sample(range(n), 2)
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
                if not satisfies_constraints(new_sol):
                    new_sol = generate_valid_solution()

            if total_cost(new_sol) == float("inf"):
                new_sol = generate_valid_solution()

            if satisfies_constraints(new_sol) and total_cost(new_sol) != float("inf"):
                new_wolves.append(new_sol)

        population = new_wolves
        print(f"Generation {gen + 1} | Minimal cost: {total_cost(alpha):.2f}")

    elapsed_time = time.time() - start_time  # Timer end
    print(f"Total processing time: {elapsed_time:.4f} seconds")
    return alpha, total_cost(alpha)




