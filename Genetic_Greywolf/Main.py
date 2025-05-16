import random
import time
import numpy as np

# === Functii comune ===
def read_sop_file(filename):
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

def extract_precedences(matrix):
    n = len(matrix)
    preds = {i: [] for i in range(n)}
    succs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == -1:
                preds[i].append(j)
                succs[j].append(i)
    return preds, succs

def preds_to_constraints(preds):
    return [(p, n) for n, pred in preds.items() for p in pred]

def is_valid(sequence, constraints):
    pos = {v: i for i, v in enumerate(sequence)}
    return all(pos[a] < pos[b] for a, b in constraints)

def fitness(route, matrix):
    return sum(matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

def repair_sequence(seq, constraints):
    pos = {n: i for i, n in enumerate(seq)}
    for a, b in constraints:
        if pos[a] > pos[b]:
            seq.remove(a)
            seq.insert(pos[b], a)
            pos = {n: i for i, n in enumerate(seq)}
    return seq

# === Genetic Algorithm ===
def generate_initial_population(pop_size, nodes, constraints):
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

def tournament_selection(population, matrix, k=3):
    return min(random.sample(population, k), key=lambda r: fitness(r, matrix))

def order_crossover(p1, p2, constraints):
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

# === Grey Wolf Optimizer ===
def gwo_sop(matrix, preds, constraints, initial_population, iterations=100):
    wolves = [repair_sequence(ind.copy(), constraints) for ind in initial_population]
    scores = [fitness(w, matrix) for w in wolves]
    idx = np.argsort(scores)
    alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
    alpha_score = scores[idx[0]]

    for t in range(1, iterations + 1):
        a = 2 - 2 * (t - 1) / iterations
        new_wolves = []
        for X in wolves:
            candidates = []
            for leader in (alpha, beta, delta):
                C = random.uniform(0, 2)
                A = a * random.uniform(-1, 1)
                D = sum(1 for i, j in zip(leader, X) if i != j)
                num_swaps = max(1, int(abs(A) * D))
                p = leader.copy()
                for _ in range(num_swaps):
                    i, j = random.sample(range(len(X)), 2)
                    p[i], p[j] = p[j], p[i]
                candidates.append(p)
            candidates.append(X)
            valid_cands = [repair_sequence(p, constraints) for p in candidates if is_valid(p, constraints)]
            best = min(valid_cands, key=lambda p: fitness(p, matrix))
            new_wolves.append(best)

        wolves = new_wolves
        scores = [fitness(w, matrix) for w in wolves]
        idx = np.argsort(scores)
        if scores[idx[0]] < alpha_score:
            alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
            alpha_score = scores[idx[0]]
    return alpha, alpha_score

# === Hibrid GA + GWO ===
def hybrid_ga_gwo(matrix, constraints, pop_size=100, ga_gens=500, gwo_iters=100):
    nodes = list(range(len(matrix)))
    population = generate_initial_population(pop_size, nodes, constraints)
    best_solution = min(population, key=lambda r: fitness(r, matrix))
    best_cost = fitness(best_solution, matrix)
    mutation_rate = 0.1

    for gen in range(ga_gens):
        population.sort(key=lambda r: fitness(r, matrix))
        elites = population[:pop_size // 10]
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

        if gen % 100 == 0:
            print(f"[GA Gen {gen}] Cost: {best_cost}")

    # 🐺 Optimizare finală cu GWO
    final_solution, final_cost = gwo_sop(matrix, extract_precedences(matrix)[0], constraints, population, gwo_iters)
    return final_solution, final_cost

if __name__ == "__main__":
    filename = "../Data/ESC47.sop"
    matrix = read_sop_file(filename)
    preds, _ = extract_precedences(matrix)
    constraints = preds_to_constraints(preds)

    print(f"Nodes: {len(matrix)} | Constraints: {len(constraints)}")
    sol, cost = hybrid_ga_gwo(matrix, constraints, pop_size=100, ga_gens=1000, gwo_iters=200)
    print("\nFinal best route:", sol)
    print("Final cost:", cost)
    print("Valid solution?", is_valid(sol, constraints))
