from scipy.optimize import minimize
import numpy as np
import itertools as it

EPS = 10 ** -16
N = 5

def greedy_worse_than_something(x):
    greedy = x[0] * max(x[5], x[3]) + x[1] * max(x[2], x[4])

    bfirst = x[2] * max(x[1], x[5]) + x[3] * max(x[0], x[4])
    cfirst = x[4] * max(x[1], x[3]) + x[5] * max(x[0], x[2])

    return max(bfirst, cfirst) - greedy - 0.001

def marg_val(x):
    first = (x[0] * x[4] + x[1] * x[3]) * (x[2] + x[5])
    second = (x[3] * x[7] + x[4] * x[6]) * (x[5] + x[8])
    third = (x[0] * x[7] + x[1] * x[6]) * (x[2] + x[8])

    return first + second + third

def one_dot(x):
    return x[2] * x[0] + x[5] * x[3] + x[8] * x[6]

def two_dot(x):
    return x[2] * x[1] + x[5] * x[4] + x[8] * x[7]

def one_two_dot(x):
    return x[0] * x[1] + x[3] * x[4] + x[6] * x[7]

def ab_max(x):
    return one_two_dot(x) - max(one_dot(x), two_dot(x))

def greater_than_both(x):
    return marg_val(x) - max(one_dot(x), two_dot(x))

# Must be valid probability values
def valid_pr_a(x): return x[0] + x[3] + x[6] - 1
def valid_pr_b(x): return x[1] + x[4] + x[7] - 1
def valid_pr_c(x): return x[2] + x[5] + x[8] - 1

def nperm(x):
    running_total = 0
    for permutation in it.permutations(range(N)):
        product = 1
        # print([ chr(ord('A') + permutation[i]) for i in range(N) ])
        for i, c in enumerate(permutation):
            # x_i = c
            product *= x[N * i + c]
        running_total += product
    
    return running_total

def covered_four(x):
    return 1 - nperm(x)

def dp_ij(x, i, j):
    running_total = 0
    for column in range(N):
        icolumn = x[N * i + column]
        jcolumn = x[N * j + column]
        running_total += icolumn * jcolumn
    
    return running_total

def max_dp(x):
    current_max = 0
    for i in range(N):
        for j in range(i + 1, N):
            current_max = max(current_max, dp_ij(x, i, j))
    
    return current_max

def valid_pr_1(x):
    return 1 - sum(x[0:4])
def valid_pr_2(x):
    return 1 - sum(x[4:8])
def valid_pr_3(x):
    return 1 - sum(x[8:12])
def valid_pr_4(x):
    return 1 - sum(x[12:16])

def four_two_diff(x):
    return max_dp(x) - (0.25 * covered_four(x))

# Constraints list
constraints = [
    # {'type': 'ineq', 'fun': greater_than_both},
    # {'type': 'ineq', 'fun': ab_max},
    # {'type': 'ineq', 'fun': four_two_diff},

    {'type': 'eq', 'fun': valid_pr_1},
    {'type': 'eq', 'fun': valid_pr_2},
    {'type': 'eq', 'fun': valid_pr_3},
    {'type': 'eq', 'fun': valid_pr_4},
]

# Initial guess
# x0 = np.array([0.25] * 16)
# x0 = np.random.rand(16)
# for row in range(4):
#     total = sum([ x0[N * row + i] for i in range(N) ])
#     x0[N*row : N*row + N] /= total

x0 = np.array([0.25, 0.25, 0.25, 0.25, 0,
               1.0, 0.0, 0.0, 0.0, 0,
               0.0, 1.0, 0.0, 0.0, 0,
               0.0, 0.0, 1.0, 0.0, 0,
               0.0, 0.0, 0.0, 1.0, 0])

# Bounds (if needed)
bounds = [(0, None)] * 16

# Dummy objective function (we just want feasibility)
# result = minimize(lambda x: four_two_diff(x), x0, method='SLSQP', bounds=bounds, constraints=constraints)

def three_perm(x):
    one = x[0] * x[4] * x[8]
    two = x[0] * x[7] * x[5]

    three = x[6] * x[1] * x[5]
    four = x[6] * x[4] * x[2]

    five = x[3] * x[7] * x[2]
    six = x[3] * x[1] * x[8]

    return one + two + three + four + five + six

def round_print(x):
    print(np.matrix.round(x, 3))

print(nperm(x0))

# if result.success:
#     x = result.x
#     print("Solution found:")
#     round_print(x[0:4])
#     round_print(x[4:8])
#     round_print(x[8:12])
#     round_print(x[12:16])

#     print(four_two_diff(x))
#     print(1 - nperm(x))
#     print(max_dp(x))
# else:
#     print("No feasible solution found.")