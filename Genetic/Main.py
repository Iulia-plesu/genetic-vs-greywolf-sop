import random
import time
import heapq

def read_sop_file_v2(filename):
    """
    Reads a SOP (Sequential Ordering Problem) file and extracts the cost matrix.

    Args:
        filename (str): Path to the SOP instance file.

    Returns:
        list[list[int]]: Cost matrix.
    """
    matrix, reading = [], False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('EDGE_WEIGHT_SECTION'):
                reading = True
                continue
            if line == 'EOF':
                break
            if reading and line:
                matrix.append(list(map(int, line.split())))
    return matrix

def extract_precedences_v2(mat):
    """
    Extracts precedence constraints from the SOP matrix.

    Args:
        mat (list[list[int]]): SOP matrix with -1 representing precedence.

    Returns:
        tuple[dict, dict]: Two dictionaries containing predecessors and successors.
    """
    n = len(mat)
    preds = {i: [] for i in range(n)}
    succs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mat[i][j] == -1:
                preds[i].append(j)
                succs[j].append(i)
    return preds, succs

def preds_to_constraints(preds):
    """
    Converts predecessor dictionary into a list of (from, to) constraints.

    Args:
        preds (dict): Predecessor dictionary.

    Returns:
        list[tuple[int, int]]: List of precedence constraints.
    """
    return [(p, n) for n, pred in preds.items() for p in pred]

def is_valid(sequence, constraints):
    """
    Validates if a given sequence satisfies the precedence constraints.

    Args:
        sequence (list[int]): Route/sequence of nodes.
        constraints (list[tuple[int, int]]): List of constraints.

    Returns:
        bool: True if sequence is valid, False otherwise.
    """
    pos = {v: i for i, v in enumerate(sequence)}
    return all(pos[a] < pos[b] for a, b in constraints)

def fitness(route, matrix):
    """
    Calculates the total cost of a given route.

    Args:
        route (list[int]): Sequence of nodes.
        matrix (list[list[int]]): Cost matrix.

    Returns:
        int: Total cost of the route.
    """
    return sum(matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

def generate_initial_population(pop_size, nodes, constraints):
    """
    Generates a valid initial population using randomized topological sorting.

    Args:
        pop_size (int): Population size.
        nodes (list[int]): List of node indices.
        constraints (list[tuple[int, int]]): Precedence constraints.

    Returns:
        list[list[int]]: List of valid sequences.
    """
    graph = {node: set() for node in nodes}
    in_degree = {node: 0 for node in nodes}
    for a, b in constraints:
        graph[a].add(b)
        in_degree[b] += 1

    population = []
    for _ in range(pop_size):
        in_deg = in_degree.copy()
        candidates = [n for n in nodes if in_deg[n] == 0]
        permutation = []
        while candidates:
            node = random.choice(candidates)
            permutation.append(node)
            candidates.remove(node)
            for neighbor in graph[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    candidates.append(neighbor)
        if len(permutation) == len(nodes):
            population.append(permutation)
        else:
            population.append(repair_sequence(random.sample(nodes, len(nodes)), constraints))
    return population

def repair_sequence(seq, constraints):
    """
    Repairs a sequence to satisfy precedence constraints using a greedy swap.

    Args:
        seq (list[int]): Node sequence.
        constraints (list[tuple[int, int]]): Constraints.

    Returns:
        list[int]: Repaired sequence.
    """
    pos = {n: i for i, n in enumerate(seq)}
    for a, b in constraints:
        if pos[a] > pos[b]:
            seq.remove(a)
            seq.insert(pos[b], a)
            pos = {n: i for i, n in enumerate(seq)}
    return seq

def tournament_selection(population, matrix, k=3):
    """
    Selects a parent using tournament selection.

    Args:
        population (list[list[int]]): Current population.
        matrix (list[list[int]]): Cost matrix.
        k (int): Tournament size.

    Returns:
        list[int]: Selected individual.
    """
    return min(random.sample(population, k), key=lambda r: fitness(r, matrix))

def order_crossover(p1, p2, constraints):
    """
    Applies Order Crossover (OX) and repairs the child to respect constraints.

    Args:
        p1 (list[int]): Parent 1.
        p2 (list[int]): Parent 2.
        constraints (list[tuple[int, int]]): Constraints.

    Returns:
        list[int]: Valid offspring.
    """
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = p1[a:b+1]

    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1

    return repair_sequence(child, constraints)

def mutate(route, constraints, mutation_rate):
    """
    Applies mutation to the route with a given probability.

    Args:
        route (list[int]): Individual to mutate.
        constraints (list[tuple[int, int]]): Constraints.
        mutation_rate (float): Probability of mutation.

    Returns:
        list[int]: Mutated individual.
    """
    if random.random() < mutation_rate:
        for _ in range(5):
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i], route[j] = route[j], route[i]
            if is_valid(route, constraints):
                return route
            route[i], route[j] = route[j], route[i]
        return repair_sequence(route, constraints)
    return route

def local_search(route, matrix, constraints):
    """
    Applies simple local search by pairwise swaps to improve fitness.

    Args:
        route (list[int]): Current route.
        matrix (list[list[int]]): Cost matrix.
        constraints (list[tuple[int, int]]): Constraints.

    Returns:
        list[int]: Locally improved route.
    """
    best = route[:]
    best_cost = fitness(best, matrix)
    for _ in range(50):
        i, j = sorted(random.sample(range(len(route)), 2))
        new_route = best[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        if is_valid(new_route, constraints):
            new_cost = fitness(new_route, matrix)
            if new_cost < best_cost:
                best, best_cost = new_route, new_cost
    return best

def genetic_algorithm(matrix, constraints, pop_size=100, generations=1500):
    """
    Runs the Genetic Algorithm to solve the SOP.

    Args:
        matrix (list[list[int]]): SOP cost matrix.
        constraints (list[tuple[int, int]]): Precedence constraints.
        pop_size (int): Population size.
        generations (int): Number of generations.

    Returns:
        tuple[list[int], float]: Best solution and its cost.
    """
    nodes = list(range(len(matrix)))
    population = generate_initial_population(pop_size, nodes, constraints)
    best_solution = min(population, key=lambda r: fitness(r, matrix))
    best_cost = fitness(best_solution, matrix)
    mutation_rate = 0.1
    start_time = time.time()

    for gen in range(generations):
        population.sort(key=lambda r: fitness(r, matrix))
        elites = population[:max(5, pop_size // 10)]

        # Increase mutation rate over time to escape local minima
        if gen % 100 == 0 and gen > 0:
            mutation_rate = min(0.5, mutation_rate + 0.05)

        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, matrix)
            p2 = tournament_selection(population, matrix)
            child = order_crossover(p1, p2, constraints)
            child = mutate(child, constraints, mutation_rate)
            new_pop.append(child)

        population = new_pop
        candidate = min(population, key=lambda r: fitness(r, matrix))
        candidate = local_search(candidate, matrix, constraints)
        cost = fitness(candidate, matrix)

        if cost < best_cost:
            best_solution = candidate
            best_cost = cost

        if gen % 20 == 0:
            print(f"Gen {gen}: Best cost = {best_cost:.2f} | Time: {time.time() - start_time:.2f}s")

    print(f"\nFinal Time: {time.time() - start_time:.2f}s")
    return best_solution, best_cost

if __name__ == "__main__":
    # Path to SOP instance file
    filename = r"..\Data\p43.1.sop"

    matrix = read_sop_file_v2(filename)
    preds, succs = extract_precedences_v2(matrix)
    constraints = preds_to_constraints(preds)

    print(f"Matrix size: {len(matrix)} x {len(matrix[0])}")
    print(f"Precedence constraints: {len(constraints)}")

    best_route, best_cost = genetic_algorithm(
        matrix,
        constraints,
        pop_size=150,
        generations=2500
    )

    print(f"\nBest route: {best_route}")
    print(f"Best cost: {best_cost}")
