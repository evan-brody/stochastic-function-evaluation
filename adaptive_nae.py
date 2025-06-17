from scipy.optimize import minimize
import numpy as np

EPS = 10 ** -16

def greedy_worse_than_something(x):
    greedy = x[0] * max(x[5], x[3]) + x[1] * max(x[2], x[4])

    bfirst = x[2] * max(x[1], x[5]) + x[3] * max(x[0], x[4])
    cfirst = x[4] * max(x[1], x[3]) + x[5] * max(x[0], x[2])

    return max(bfirst, cfirst) - greedy - 0.001


# Must be valid probability values
def valid_pr_a(x): return 1 - x[0] - x[1]
def valid_pr_b(x): return 1 - x[2] - x[3]
def valid_pr_c(x): return 1 - x[4] - x[5]

# a is assumed to have the greatest value of Pr[G] + Pr[B]
def asum_greater_bsum(x): return x[0] + x[1] - x[2] - x[3] - 0.001
def asum_greater_csum(x): return x[0] + x[1] - x[4] - x[5] - 0.001

# def abeta_greater_bbeta(x): return x[0] - x[2]
# def abeta_greater_cbeta(x): return x[0] - x[4]
# def agamma_greater_bgamma(x): return x[1] - x[3]
# def agamma_greater_cgamma(x): return x[1] - x[5]

# def bbeta_greater_cbeta(x): return x[2] - x[4]
# def cgamma_greater_bgamma(x): return x[5] - x[3]

# Constraints list
constraints = [
    {'type': 'ineq', 'fun': greedy_worse_than_something},

    {'type': 'ineq', 'fun': valid_pr_a},
    {'type': 'ineq', 'fun': valid_pr_b},
    {'type': 'ineq', 'fun': valid_pr_c},

    {'type': 'ineq', 'fun': asum_greater_bsum},
    {'type': 'ineq', 'fun': asum_greater_csum}
]

# Initial guess
x0 = np.array([0.4, 0.4, 0.3, 0.1, 0.1, 0.3])

# Bounds (if needed)
bounds = [(0, None)] * 6

# Dummy objective function (we just want feasibility)
result = minimize(lambda _: 0, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x = result.x
    print("Feasible solution found:")
    print(f"x0={x[0]:.4f}, x1={x[1]:.4f}")
    print(f"x2={x[2]:.4f}, x3={x[3]:.4f}")
    print(f"x4={x[4]:.4f}, x5={x[5]:.4f}")

    greedy = x[0] * x[5] + x[1] * x[2]

    bfirst = x[2] * x[1] + x[3] * x[0]
    cfirst = x[4] * x[1] + x[5] * x[0]
    print(f"Greedy: {greedy}")
    print(f"bfirst: {bfirst}")
    print(f"cfirst: {cfirst}")
else:
    print("No feasible solution found.")