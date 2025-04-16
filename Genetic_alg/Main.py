import random
import time


def read_sop_file(path):
    """
    Reads an SOP file and returns the distance matrix and list of precedence constraints.

    Parameters:
        path (str): The path to the SOP file.

    Returns:
        tuple: (distance_matrix, precedence_constraints)
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    size = 0
    distance_matrix = []
    edge_section_started = False


    all_weights = []
    for line in lines:
        line = line.strip()

        if line.startswith("DIMENSION"):
            size = int(line.split(":")[1].strip())

        elif line.startswith("EDGE_WEIGHT_SECTION"):
            edge_section_started = True

        elif edge_section_started:
            if line == "EOF":
                break
            parts = line.split()
            row = [1000000 if int(x) == -1 else int(x) for x in parts]
            all_weights.extend(row)


    for i in range(size):
        start = i * size
        end = start + size
        distance_matrix.append(all_weights[start:end])

    precedence_constraints = []

    return distance_matrix, precedence_constraints


def is_valid(sequence, precedence_constraints):
    """
    Checks if the given sequence respects the precedence constraints.

    Parameters:
        sequence (list): The sequence of nodes.
        precedence_constraints (list): List of (a, b) tuples where a must come before b.

    Returns:
        bool: True if the sequence is valid; otherwise False.
    """
    return all(sequence.index(a) < sequence.index(b) for (a, b) in precedence_constraints)


def generate_initial_population(pop_size, nodes, precedence_constraints):
    """
    Generates the initial population for the genetic algorithm.

    Parameters:
        pop_size (int): The size of the population.
        nodes (list): List of nodes.
        precedence_constraints (list): Precedence constraints.

    Returns:
        list: A list of valid permutations of nodes that satisfy the constraints.
    """
    population = []
    while len(population) < pop_size:
        perm = random.sample(nodes, len(nodes))
        if is_valid(perm, precedence_constraints):
            population.append(perm)
    return population


def fitness(route, distance_matrix):
    """
    Calculates the total cost of a route using the provided distance matrix.

    Parameters:
        route (list): The route (sequence of nodes).
        distance_matrix (list): Matrix containing the distances between nodes.

    Returns:
        int: The cost of the route.
    """
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    return cost


def tournament_selection(population, distance_matrix, k=3):
    """
    Selects an individual from the population using the tournament method.

    Parameters:
        population (list): The current population.
        distance_matrix (list): The distance matrix.
        k (int): Number of individuals to participate in the tournament.

    Returns:
        list: The best route from the tournament selection.
    """
    selected = random.sample(population, k)
    selected.sort(key=lambda r: fitness(r, distance_matrix))
    return selected[0]


def order_crossover(parent1, parent2, precedence_constraints):
    """
    Applies order crossover (OX) on two parents and returns a repaired child.

    Parameters:
        parent1 (list): The first parent.
        parent2 (list): The second parent.
        precedence_constraints (list): Precedence constraints used for repair.

    Returns:
        list: The child sequence after crossover and repair.
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child_p1 = parent1[a:b + 1]
    child_p2 = [item for item in parent2 if item not in child_p1]
    child = child_p2[:a] + child_p1 + child_p2[a:]
    return repair_sequence(child, precedence_constraints)


def mutate(route, precedence_constraints, mutation_rate=0.1):
    """
    Applies mutation on a route with a given mutation probability and repairs the sequence.

    Parameters:
        route (list): The route to be mutated.
        precedence_constraints (list): Precedence constraints for repair.
        mutation_rate (float): The probability of mutation.

    Returns:
        list: The mutated and repaired route.
    """
    if random.random() < mutation_rate:
        a, b = sorted(random.sample(range(len(route)), 2))
        route[a], route[b] = route[b], route[a]
        route = repair_sequence(route, precedence_constraints)
    return route


def repair_sequence(route, precedence_constraints):
    """
    Repairs a sequence to ensure that all precedence constraints are met.

    Parameters:
        route (list): The sequence that requires repair.
        precedence_constraints (list): List of precedence constraints.

    Returns:
        list: The repaired sequence.
    """
    for (a, b) in precedence_constraints:
        if route.index(a) > route.index(b):
            route.remove(a)
            index_b = route.index(b)
            route.insert(index_b, a)
    return route


def genetic_algorithm(distance_matrix, precedence_constraints, pop_size=50, generations=200):
    """
    Executes a genetic algorithm to find the optimal route based on the distance matrix.

    Parameters:
        distance_matrix (list): The matrix of distances between nodes.
        precedence_constraints (list): Precedence constraints.
        pop_size (int): The population size.
        generations (int): The number of generations to evolve.

    Returns:
        tuple: (best_solution, best_cost)
    """
    nodes = list(range(len(distance_matrix)))
    population = generate_initial_population(pop_size, nodes, precedence_constraints)
    best_solution = min(population, key=lambda r: fitness(r, distance_matrix))
    start_time = time.time()

    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, distance_matrix)
            parent2 = tournament_selection(population, distance_matrix)
            child = order_crossover(parent1, parent2, precedence_constraints)
            child = mutate(child, precedence_constraints)
            new_population.append(child)

        population = new_population

        gen_best = min(population, key=lambda r: fitness(r, distance_matrix))
        if fitness(gen_best, distance_matrix) < fitness(best_solution, distance_matrix):
            best_solution = gen_best

        if gen % 10 == 0:
            print(
                f"Gen {gen}: Best Cost = {fitness(best_solution, distance_matrix)} | Time: {time.time() - start_time:.2f}s")
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

    return best_solution, fitness(best_solution, distance_matrix)


if __name__ == "__main__":
    path = r"..\Data\ESC47.sop"
    distance_matrix, precedence_constraints = read_sop_file(path)

    print(f"Matrix: {len(distance_matrix)} x {len(distance_matrix[0])}")
    print(f"Number of constraints: {len(precedence_constraints)}")

    best_route, best_cost = genetic_algorithm(distance_matrix, precedence_constraints)
    print(f"\nBest Route: {best_route}")
    print(f"Number of elements: {len(best_route)}")
    print(f"Cost: {best_cost}")
