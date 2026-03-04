# @file     aag.py
# @author   Evan Brody
# @brief    For testing conjectures related to the additive adaptivity gap of the Unanimous Vote problem.

import numpy as np
import copy

N = 10

def greedy_dpn(p):
    heads_sum = 0.0
    current_term = p[0]
    # index n - 3 = p_{n-2} will be the last term added
    for i in range(N - 1): # from 1 through n - 2
        heads_sum += current_term 
        current_term *= p[i]
    
    tails_sum = 0.0
    current_term = 1 - p[0]
    for i in range(N - 1):
        tails_sum -= current_term
        current_term *= 1 - p[i]

    return heads_sum + tails_sum

def opt_dpn(p):
    total = 0.0
    current_term = 1 - p[0]
    # starts with index N - 2 = p_{n - 1}
    # last term added is 4 = p_3, because p_2 is the missing factor
    for i in range(N - 2, 2, -1): # n - 1 down to 3
        total += current_term
        current_term *= 1 - p[i]

    return total

def dpn(p):
    return greedy_dpn(p) + opt_dpn(p)

def greedy_dp1(p):
    heads_sum = 0.0
    current_term = p[N - 1]
    for i in range(N - 1, 2, -1):
        heads_sum += current_term
        current_term *= p[i]
    
    tails_sum = 0.0
    current_term = 1 - p[N - 1]
    for i in range(N - 1, 2, -1):
        tails_sum -= current_term
        current_term *= 1 - p[i]
    
    return heads_sum + tails_sum

def opt_dp1(p):
    total = 0.0
    current_term = p[N - 1]
    for i in range(2, N): # last term to. be added is p_{n-2} = n - 1
        total -= current_term
        current_term *= p[i]
    
    return total

def dp1(p):
    return greedy_dp1(p) + opt_dp1(p)

def test_values(p):
    this_dp1 = dp1(p)
    this_dpn = dpn(p)

    if this_dp1 <= 0 or this_dpn >= 0:
        return -1
    
    return this_dp1 - this_dpn

def get_scale_vector():
    scale_vector = np.ones((N,), dtype=float)
    scale_vector += np.random.normal(scale=0.01)

    return scale_vector

def normalize(vector):
    sv = sum(vector)
    vector[:] /= sv

    return vector

def clamp(vector):
    for j in range(len(vector)):
        vector[j] = min(1.0, vector[j])
    
    return vector
    
def get_child_values(p):
    new_p = copy.deepcopy(p)
    scale_vector = get_scale_vector()
    new_p *= scale_vector
    new_p = normalize(clamp(new_p))

    return new_p

def get_p():
    p = np.random.rand(N)
    p.sort()

    return p

GENERATION_SIZE = 1000
GENERATION_COUNT = 100_000
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_diff_instance = None
    try:
        for _ in range(1_00_000):
            p = get_p()

            diff = test_values(p)
            if diff > max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(p )

            if i % 1000 == 0:
                print(f"-------------[{i} -> {round(max_diff, 5)}]-------------")
            
            i += 1
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                p = get_child_values(current_parent)

                diff = test_values(p)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(p)

                if i % 1000 == 0:
                    print(f"-------------[gen {_}, {i} -> {round(max_diff, 5)}]-------------")
                
                i += 1

        print(p); print()
        print(f"max diff: {max_diff}"); print()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        print(p); print()
        print(f"max diff: {max_diff}"); print()