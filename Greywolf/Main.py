import random
import time


def read_data_from_file(file_path):
    """
    Reads an SOP file and extracts the dimension, cost matrix, and precedence constraints.

    The function processes the file to extract the "DIMENSION" and "EDGE_WEIGHT_SECTION" data.
    It interprets a value of -1 as an infinite cost (represented by float('inf')). Precedence constraints
    are currently returned as an empty list.

    Parameters:
        file_path (str): The path to the input SOP file.

    Returns:
        tuple: A tuple containing:
            - n (int): The number of nodes (dimension).
            - cost (list of float): The cost matrix.
            - precedence (list): The list of precedence constraints.
    """
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


def satisfies_constraints(chromosome):
    """
    Checks if a given chromosome (route/solution) respects all precedence constraints.

    This function uses a global precedence dictionary (precedence_dict) where each key is a node
    that has constraints and the value is the set of antecedent nodes that must precede it.

    Parameters:
        chromosome (list): The sequence (list of nodes) representing a solution.

    Returns:
        bool: True if the sequence satisfies all precedence constraints, False otherwise.
    """
    positions = {node: idx for idx, node in enumerate(chromosome)}
    for b, antecedents in precedence_dict.items():
        b_pos = positions[b]
        for a in antecedents:
            if positions[a] > b_pos:
                return False
    return True


def calculate_cost(chromosome):
    """
    Calculates the total cost of a given chromosome (route).

    It sums the costs between consecutive nodes using the global cost matrix.
    If an edge has an infinite cost, the function returns float('inf') immediately.

    Parameters:
        chromosome (list): The sequence of nodes representing the route.

    Returns:
        float: The total cost of the route, or float('inf') if the route is invalid.
    """
    total = 0
    for i in range(len(chromosome) - 1):
        current_cost = cost[chromosome[i]][chromosome[i + 1]]
        if current_cost == float('inf'):
            return float('inf')
        total += current_cost
    return total or float('inf')


def generate_valid_solution(max_attempts=1000):
    """
    Attempts to generate a valid solution (chromosome) that satisfies the precedence constraints.

    It makes up to max_attempts of random shuffles of the node list until a valid solution is found.
    If a valid solution is found, it returns the solution along with its calculated cost.

    Parameters:
        max_attempts (int): The maximum number of attempts to generate a valid solution.

    Returns:
        tuple: (solution (list), solution_cost (float)); returns (None, float('inf')) if no valid
               solution is found after max_attempts.
    """
    for _ in range(max_attempts):
        solution = list(range(n))
        random.shuffle(solution)
        if satisfies_constraints(solution):
            return solution, calculate_cost(solution)
    print("Max attempts reached generating valid solution")
    return None, float('inf')


def combine(parents):
    """
    Combines parent solutions into a new solution using a set-based approach.

    The function traverses the parents' solutions position by position to select nodes that have not
    yet been added to the new solution. Any missing nodes are appended after a shuffle.
    The resulting new solution is then checked against the constraints and, if needed, repaired by generating
    a new valid solution.

    Parameters:
        parents (list): A list of parent solutions (each is a list of nodes).

    Returns:
        tuple: (new_solution (list), new_solution_cost (float))
    """
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


def gwo_sop(num_wolves=35, max_iter=200):
    """
    Implements the Grey Wolf Optimizer (GWO) for solving the Sequential Ordering Problem (SOP).

    The function initializes a population of candidate solutions (wolves) and iteratively generates new
    populations by combining and mutating the best solutions. It outputs the best found solution and its cost.

    Parameters:
        num_wolves (int): The number of candidate solutions (wolves) in the population.
        max_iter (int): The maximum number of iterations (generations) to run the optimizer.

    Returns:
        tuple: (best_solution (list), best_cost (float))
    """
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
        if gen % 10 == 0:
            print(f"Gen {gen} | Best Cost: {best_cost:.1f} | Time: {time.time() - start_time:.2f}s")

    best_solution, best_cost = min(population, key=lambda x: x[1])
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
    return best_solution, best_cost


if __name__ == "__main__":
    file_path = '../Data/ESC47.sop'
    n, cost, precedence = read_data_from_file(file_path)
    print("Number of constraints:", len(precedence))

    precedence_dict = {b: {a for a, b_p in precedence if b_p == b} for b in set(b for _, b in precedence)}

    solution, final_cost = gwo_sop()

    print("\nOptimal solution:", solution)
    print(f"Number of elements: {len(solution)}")
    print("Total cost:", final_cost)
