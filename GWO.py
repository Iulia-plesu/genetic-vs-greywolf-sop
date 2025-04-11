import random
n=4
cost = [
    [0, 5, 0, 9],
    [0, 0, 7, 0],
    [0, 0, 0, 2],
    [0, 0, 0, 0]
]

precedence = [(0, 2), (1, 3)]

def valid_constraint(chromosome):
    poz = {gene: idx for idx, gene in enumerate(chromosome)}
    for a, b in precedence:
        if poz[a] > poz[b]:
            return False
    return True


def total_cost(chromosome):
    total = 0
    for i in range(len(chromosome) - 1):
        if cost[chromosome[i]][chromosome[i + 1]] == float("inf"):
            return float("inf")  #return inf if we have an invalid connexion
        total += cost[chromosome[i]][chromosome[i + 1]]
    return total if total != 0 else float("inf")


def fitness(chromosome):
    return 1 / (total_cost(chromosome) + 1e-6)

def generate_valid_solution(max_attempts=1000):
    attempts = 0
    while attempts < max_attempts:
        p = list(range(n))
        random.shuffle(p)
        if valid_constraint(p):
            return p
        else:
            attempts += 1
            print(f"The generated solution is not valid.: {p}, test {attempts}/{max_attempts}")

    print("Maximum number of attempts reached to generate a valid solution.")
    return None


