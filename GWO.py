import random
import time



def read_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n = 0
    cost = []
    precedence = []
    found_dimension = False


    data_section = False
    cost_data = []
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
            if i != j and idx < len(cost_data):
                val = cost_data[idx]
                cost[i][j] = float('inf') if val == -1 else val
                idx += 1

    return n, cost, precedence


file_path = './Data/br17.10.sop'
n, cost, precedence = read_data_from_file(file_path)


precedence_dict = {b: {a for a, b_p in precedence if b_p == b} for b in set(b for _, b in precedence)}


def satisfies_constraints(chromosome):
    """Optimized constraint checking using precedence dictionary"""
    positions = {node: idx for idx, node in enumerate(chromosome)}
    for b, antecedents in precedence_dict.items():
        b_pos = positions[b]
        for a in antecedents:
            if positions[a] > b_pos:
                return False
    return True


def calculate_cost(chromosome):
    """Optimized cost calculation with early exit"""
    total = 0
    for i in range(len(chromosome) - 1):
        current_cost = cost[chromosome[i]][chromosome[i + 1]]
        if current_cost == float('inf'):
            return float('inf')
        total += current_cost
    return total or float('inf')


def generate_valid_solution(max_attempts=1000):
    """Optimized valid solution generator with cost caching"""
    for _ in range(max_attempts):
        solution = list(range(n))
        random.shuffle(solution)
        if satisfies_constraints(solution):
            return solution, calculate_cost(solution)
    print("Max attempts reached generating valid solution")
    return None, float('inf')


def combine(parents):
    """Optimized combination using set operations"""
    new_solution = []
    seen = set()
    for i in range(n):
        for parent in parents:
            node = parent[i]
            if node not in seen:
                new_solution.append(node)
                seen.add(node)
                break


    missing = list(set(range(n)) - seen)
    random.shuffle(missing)
    new_solution += missing

    if satisfies_constraints(new_solution):
        return new_solution, calculate_cost(new_solution)
    return generate_valid_solution()


def gwo_sop(num_wolves=35, max_iter=100):
    population = []
    for _ in range(num_wolves):
        sol, c = generate_valid_solution()
        if sol is not None:
            population.append((sol, c))

    if not population:
        return None, float('inf')

    start_time = time.time()

    for gen in range(max_iter):
        population.sort(key=lambda x: x[1])
        alpha, beta, delta = population[:3]
        best_cost = alpha[1]


        new_population = [alpha, beta, delta]
        while len(new_population) < num_wolves:
            # Combine parents
            new_sol, new_cost = combine([alpha[0], beta[0], delta[0]])


            if random.random() < 0.3 and new_cost != float('inf'):
                i, j = random.sample(range(n), 2)
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
                if satisfies_constraints(new_sol):
                    new_cost = calculate_cost(new_sol)
                else:
                    new_sol, new_cost = generate_valid_solution()

            if new_cost != float('inf'):
                new_population.append((new_sol, new_cost))

        population = new_population[:num_wolves]
        print(f"Gen {gen + 1:03d} | Best: {best_cost:.1f} | Time: {time.time() - start_time:.2f}s")

    best_solution, best_cost = min(population, key=lambda x: x[1])
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
    return best_solution, best_cost


solution, final_cost = gwo_sop()

print("\nOptimal solution:", solution)
print("Total cost:", final_cost)