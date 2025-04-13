import random

def read_sop_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    size = 0
    distance_matrix = []
    edge_section_started = False

    # Parcurgem fiecare linie
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
            # filtrăm valorile -1, le transformăm în ceva mare (cost infinit)
            row = [1000000 if int(x) == -1 else int(x) for x in parts]
            all_weights.extend(row)

    # transformăm în matrice pătrată
    for i in range(size):
        start = i * size
        end = start + size
        distance_matrix.append(all_weights[start:end])

    # în fișierul tău lipsesc constrângerile — le lăsăm goale
    precedence_constraints = []

    return distance_matrix, precedence_constraints


def is_valid(sequence, precedence_constraints):
    return all(sequence.index(a) < sequence.index(b) for (a, b) in precedence_constraints)


def generate_initial_population(pop_size, nodes, precedence_constraints):
    population = []
    while len(population) < pop_size:
        perm = random.sample(nodes, len(nodes))
        if is_valid(perm, precedence_constraints):
            population.append(perm)
    return population


def fitness(route, distance_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    return cost


def tournament_selection(population, distance_matrix, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda r: fitness(r, distance_matrix))
    return selected[0]


def order_crossover(parent1, parent2, precedence_constraints):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child_p1 = parent1[a:b + 1]
    child_p2 = [item for item in parent2 if item not in child_p1]
    child = child_p2[:a] + child_p1 + child_p2[a:]
    return repair_sequence(child, precedence_constraints)


def mutate(route, precedence_constraints, mutation_rate=0.1):
    if random.random() < mutation_rate:
        a, b = sorted(random.sample(range(len(route)), 2))
        route[a], route[b] = route[b], route[a]
        route = repair_sequence(route, precedence_constraints)
    return route


def repair_sequence(route, precedence_constraints):
    for (a, b) in precedence_constraints:
        if route.index(a) > route.index(b):
            route.remove(a)
            index_b = route.index(b)
            route.insert(index_b, a)
    return route


def genetic_algorithm(distance_matrix, precedence_constraints, pop_size=50, generations=200):
    nodes = list(range(len(distance_matrix)))
    population = generate_initial_population(pop_size, nodes, precedence_constraints)
    best_solution = min(population, key=lambda r: fitness(r, distance_matrix))

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
            print(f"Gen {gen}: Best Cost = {fitness(best_solution, distance_matrix)}")

    return best_solution, fitness(best_solution, distance_matrix)


if __name__ == "__main__":
    path = r"..\Data\ESC47.sop"
    distance_matrix, precedence_constraints = read_sop_file(path)

    print(f"Matrice: {len(distance_matrix)} x {len(distance_matrix[0])}")
    print(f"Nr. constrângeri de precedență: {len(precedence_constraints)}")

    best_route, best_cost = genetic_algorithm(distance_matrix, precedence_constraints)
    print(f"\nBest Route: {best_route}")
    print(f"Cost: {best_cost}")


