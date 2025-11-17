import numpy as np
import itertools as it
import math

n = 5
x = np.random.rand(n, n)
for row in range(n):
    total = sum(x[row])
    x[row, :] /= total

def uncovered_by_S(S):
    k = len(S)
    total = 0
    for C in it.combinations(range(n), k):
        for P in it.permutations(S):
            product = 1
            for i, die in enumerate(P):
                color = C[i]
                product *= x[die, color]
            total += product
    
    return total

def covered_by_S(S):
    return 1 - uncovered_by_S(S)

def max_cover_k(k):
    max_comb = None
    max_comb_covered = 0
    for combination in it.combinations(range(n), k):
        this_combination_covered = covered_by_S(combination)
        if this_combination_covered > max_comb_covered:
            max_comb_covered = this_combination_covered
            max_comb = combination

    return set(max_comb), round(float(max_comb_covered), 4)

checked = 0
while True:
    max_i = None
    max_i_score = 0
    for i in range(2, n):
        S, uS = max_cover_k(i)
        if uS / i > max_i_score:
            max_i_score = uS / i
            max_i = i
    
    if max_i != 3:
        for i in range(2, n):
            S, uS = max_cover_k(i)
            print(S, uS / i)
        while True: pass
    
    x = np.random.rand(n, n)
    for row in range(n):
        total = sum(x[row])
        x[row, :] /= total

    checked += 1
    print(checked)