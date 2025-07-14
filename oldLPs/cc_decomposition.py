from scipy.optimize import minimize
import numpy as np

EPS = 10 ** -16

def rr_worse(x):
    t = 3
    x3 = 1 - x[0] - x[1]

    lhs = (x3 + x[0] * x[1]) ** t
    lhs += (x[1] + x[0] * x3) ** t
    lhs += (x[0] + x[1] * x3) ** t

    rhs = (x[0] ** t) + (x[1] ** t) + (x3 ** t)
    rhs += (x[0] * x[1] + x[0] * x3 + x[1] * x3) ** t

    # rhs - lhs >= 0 for a counterexample
    return rhs - lhs - 0.0001

def valid_pr(x): return 1 - x[0] - x[1]
def valid_pr1(x): return x[0] - EPS
def valid_pr2(x): return x[1] - EPS

# Constraints list
constraints = [
    {'type': 'ineq', 'fun': rr_worse},
    {'type': 'ineq', 'fun': valid_pr},
    {'type': 'ineq', 'fun': valid_pr1},
    {'type': 'ineq', 'fun': valid_pr2},
]

# Initial guess
x0 = np.array([0.1, 0.4])

# Bounds (if needed)
bounds = [(0, None)] * 2

# Dummy objective function (we just want feasibility)
result = minimize(lambda _: 0, x0, method='SLSQP', bounds=bounds, constraints=constraints)

if result.success:
    x = result.x
    print("Feasible solution found:")
    print(f"x0={x[0]}, x1={x[1]}, x2={1 - x[0] - x[1]}")
    print(f"res={rr_worse(x)}")
else:
    print("No feasible solution found.")