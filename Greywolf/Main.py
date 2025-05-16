import numpy as np
import random
import time

# --- 1. Read SOP file into a cost matrix ---
def read_sop_file(filename):
    """Reads an SOP instance file and extracts the cost matrix."""
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

# --- 2. Extract precedence and successor constraints from matrix ---
def extract_precedences(mat):
    """Builds precedence and successor lists based on -1 entries in the matrix."""
    n = len(mat)
    preds = {i: [] for i in range(n)}
    succs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mat[i][j] == -1:
                preds[i].append(j)
                succs[j].append(i)
    return preds, succs

# --- 3. Generate initial solution using random topological sort ---
def random_topo(preds, succs):
    """Generates a random topological sort that respects precedence constraints."""
    n = len(preds)
    in_deg = {i: len(preds[i]) for i in preds}
    Q = [i for i, d in in_deg.items() if d == 0]
    order = []
    while Q:
        u = random.choice(Q)
        Q.remove(u)
        order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                Q.append(v)
    return order

# --- 4. Check if a permutation is valid (respects precedences) and compute cost ---
def is_valid(perm, preds):
    """Checks if a permutation respects all precedence constraints."""
    pos = {u: i for i, u in enumerate(perm)}
    for i, ps in preds.items():
        for j in ps:
            if pos[j] >= pos[i]:
                return False
    return True

def cost_of(perm, mat):
    """Computes the total cost of a permutation based on the cost matrix."""
    return sum(mat[perm[i]][perm[i + 1]] for i in range(len(perm) - 1))

# --- 5. Repair an invalid permutation using local reordering ---
def repair(perm, preds):
    """Fixes a permutation to make it valid by reordering based on precedences."""
    pos = {u: i for i, u in enumerate(perm)}
    for i, ps in preds.items():
        for j in ps:
            if pos[j] >= pos[i]:
                perm.insert(pos[i], perm.pop(pos[j]))
                pos = {u: k for k, u in enumerate(perm)}
    return perm

# --- 6. Generate candidate solutions by applying multiple mutation types ---
def mutate_candidates(base, num_swaps):
    """Generates new permutations using various mutation operations."""
    n = len(base)
    cands = []

    # Swap two elements
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        p = base.copy()
        p[i], p[j] = p[j], p[i]
        cands.append(p)

    # Insertion mutation
    i, j = random.sample(range(n), 2)
    p = base.copy()
    u = p.pop(i)
    p.insert(j, u)
    cands.append(p)

    # Reversal mutation
    i, j = sorted(random.sample(range(n), 2))
    p = base.copy()
    p[i:j + 1] = reversed(p[i:j + 1])
    cands.append(p)

    return cands

# --- 7. Grey Wolf Optimizer with timing per epoch and total time ---
def gwo_sop_with_timing(matrix, preds, succs, num_wolves=100, max_iter=300, num_runs=3):
    """
    Applies the Grey Wolf Optimizer to the SOP problem.
    Tracks and prints timing for each run and iteration.
    """
    all_scores = []
    all_perms = []

    total_start = time.perf_counter()
    for run in range(1, num_runs + 1):
        print(f"\nRun {run}/{num_runs}")
        wolves = [repair(random_topo(preds, succs), preds) for _ in range(num_wolves)]
        scores = [cost_of(w, matrix) for w in wolves]
        idx = np.argsort(scores)
        alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
        alpha_score = scores[idx[0]]

        for t in range(1, max_iter + 1):
            epoch_start = time.perf_counter()
            a = 2 - 2 * (t - 1) / max_iter
            new_wolves = []
            for X in wolves:
                candidates = []
                for leader in (alpha, beta, delta):
                    C = random.uniform(0, 2)
                    A = a * random.uniform(-1, 1)
                    D = sum(1 for i, j in zip(leader, X) if i != j)
                    num_swaps = max(1, int(abs(A) * D))
                    for _ in range(num_swaps):
                        i, j = random.sample(range(len(X)), 2)
                        p = leader.copy()
                        p[i], p[j] = p[j], p[i]
                        candidates.append(p)

                candidates.append(X)
                valid_cands = [repair(p.copy(), preds) for p in candidates if is_valid(repair(p.copy(), preds), preds)]
                best = min(valid_cands, key=lambda p: cost_of(p, matrix))
                new_wolves.append(best)

            wolves = new_wolves
            scores = [cost_of(w, matrix) for w in wolves]
            idx = np.argsort(scores)
            if scores[idx[0]] < alpha_score:
                alpha, beta, delta = wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
                alpha_score = scores[idx[0]]

            epoch_time = time.perf_counter() - epoch_start
            print(f"Epoch {t}/{max_iter} - Best cost: {alpha_score} - Time: {epoch_time:.4f}s")

        all_scores.append(alpha_score)
        all_perms.append(alpha)

    total_time = time.perf_counter() - total_start
    best_run = np.argmin(all_scores)
    average_score = np.mean(all_scores)
    best_score = all_scores[best_run]
    best_perm = all_perms[best_run]

    print(f"\nTotal runs time: {total_time:.2f}s")
    return average_score, best_score, best_perm

# --- Example usage ---
if __name__ == '__main__':
    mat = read_sop_file('../Data/p43.1.sop')
    preds, succs = extract_precedences(mat)
    avg, best_score, best_perm = gwo_sop_with_timing(mat, preds, succs)
    print('Avg:', avg)
    print('Best:', best_score)
    print('Perm:', best_perm)
