import random
import time


def read_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n = 0
    cost_data = []
    found_dimension = False
    data_section = False
    for line in lines:
        line = line.strip()
        if not found_dimension and line.startswith("DIMENSION"):
            n = int(line.split(":")[1])
            found_dimension = True
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            data_section = True
            continue
        elif line == "EOF":
            break
        elif data_section:
            cost_data.extend(map(int, line.split()))

    cost = [[float('inf')] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                val = cost_data[idx]
                cost[i][j] = float('inf') if val == -1 else val
                idx += 1

    return n, cost, []


def satisfies_constraints(chromosome):
    pos = {node: idx for idx, node in enumerate(chromosome)}
    for b, preds in precedence_dict.items():
        if any(pos[a] > pos[b] for a in preds):
            return False
    return True


def calculate_cost(chromosome):
    total = 0
    for i in range(len(chromosome) - 1):
        c = cost[chromosome[i]][chromosome[i + 1]]
        if c == float('inf'):
            return float('inf')
        total += c
    return total


def generate_valid_solution():
    attempts = 0
    while attempts < 1000:
        sol = list(range(n))
        random.shuffle(sol)
        if satisfies_constraints(sol):
            return sol, calculate_cost(sol)
        attempts += 1
    return None, float('inf')


def combine(parents):
    seen = set()
    child = []
    for i in range(n):
        for p in parents:
            if p[i] not in seen:
                child.append(p[i])
                seen.add(p[i])
                break
    remaining = list(set(range(n)) - seen)
    random.shuffle(remaining)
    child.extend(remaining)
    return (child, calculate_cost(child)) if satisfies_constraints(child) else generate_valid_solution()


def mutate(solution):
    i, j = random.sample(range(n), 2)
    solution[i], solution[j] = solution[j], solution[i]
    return (solution, calculate_cost(solution)) if satisfies_constraints(solution) else generate_valid_solution()


def hybrid_ga_gwo(num_wolves=50, max_iter=250, mutation_rate=0.3):
    population = [generate_valid_solution() for _ in range(num_wolves)]
    population = [p for p in population if p[0] is not None]

    if not population:
        return None, float('inf')

    start_time = time.time()

    for gen in range(max_iter):
        population.sort(key=lambda x: x[1])
        alpha, beta, delta = population[:3]

        new_population = [alpha, beta, delta]
        while len(new_population) < num_wolves:
            if random.random() < 0.5:
                parents = random.sample(population[:10], 2)
                child, cost_child = combine([p[0] for p in parents])
            else:
                child, cost_child = combine([alpha[0], beta[0], delta[0]])

            if random.random() < mutation_rate:
                child, cost_child = mutate(child)

            if cost_child != float('inf'):
                new_population.append((child, cost_child))

        population = new_population[:num_wolves]

        if gen % 10 == 0 or gen == max_iter - 1:
            print(f"Gen {gen:3} | Best Cost: {population[0][1]:.1f} | Time: {time.time() - start_time:.2f}s")

    best_solution = min(population, key=lambda x: x[1])
    print(f"\nFinal Cost: {best_solution[1]:.1f} | Total Time: {time.time() - start_time:.2f}s")
    return best_solution


if __name__ == "__main__":
    file_path = '../Data/ESC47.sop'
    n, cost, precedence = read_data_from_file(file_path)

    precedence_dict = {b: set() for _, b in precedence}
    for a, b in precedence:
        precedence_dict[b].add(a)

    print("Număr de constrângeri:", len(precedence_dict))

    sol, final_cost = hybrid_ga_gwo()

    print("\nSoluție optimă:", sol)
    print("Cost total:", final_cost)
