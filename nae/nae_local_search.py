# @file     nae_local_search.py
# @author   Evan Brody
# @brief    Tests the local convergence rate of the NAE function.

import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpmath import mp

# Configure mp
mp.dps = 200    # Decimal places used by mp.mpf
mp.pretty = True # Turn pretty-printing on

# We use mp's floating point type
NAE_float = mp.mpf

class NAE:
    RNG = np.random.default_rng()
    PRINT_STEP = 10_000

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.init_distribution()

        # Start from a random strategy
        self.shuffle_strat()
    
    # Randomizes the strategy and updates the expected costs
    def shuffle_strat(self):
        self.strategy = self.RNG.permutation(self.n)
        self.update_expected_cost()
    
    # Generates a distribution over the tests and colorss
    def init_distribution(self):
        # Generates random "markers" in [0, 1]
        # Doing it this way is necessary because we use the custom type NAE_float
        self.distribution = np.empty(shape=(self.n, self.d), dtype=NAE_float)
        for i, c in itertools.product(range(self.n), range(self.d)):
            self.distribution[i, c] = NAE_float(np.random.rand())

        self.distribution[:, -1] = NAE_float(1)
        self.distribution.sort(axis=1)

        # Calculate the space between markers, which will be the
        # probability of some outcome
        for die in self.distribution:
            for i in range(self.d - 1, 0, -1):
                die[i] -= die[i - 1]

    # Fully recalculates the expected cost of the current strategy
    # O(dn), use only when necessary
    def update_expected_cost(self):
        self.cost = NAE_float(1)
        pr_only_seen = np.array([ p for p in self.distribution[self.strategy[0]] ], dtype=NAE_float)
        for test in self.strategy[1:]:
            self.cost += sum(pr_only_seen)
            for c, p in enumerate(self.distribution[test]):
                pr_only_seen[c] *= p

    # Performs one step of local search. The neighbourhood of the strategy
    # is defined as the set of strategies that can be formed swapping two
    # tests in the current strategy.
    # Returns true if a better strategy is found in the neighbourhood, false otherwise
    def local_step(self):
        best_swap = None
        # Improvement should be greater than machine epsilon.
        # This is to prevent a bug where local search will continually make and undo
        # a swap, each time "gaining" an improvement on the order of 10^-17 due to precision error
        best_swap_improvement = mp.eps

        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                improvement = self.gain_from_swap(i, j)
                if improvement > best_swap_improvement:
                    best_swap_improvement = improvement
                    best_swap = (i, j)
        
        if best_swap is not None:
            # Important that tables are updated before the swap instead of after
            # since the code within assumes the strategy hasn't been updated yet
            self.update_tables_after_swap(*best_swap)
            self.swap_tests(*best_swap)
            self.cost -= best_swap_improvement
            return True
        
        return False
    
    # Exactly the above function, but a strategy's neighbourhood is defined
    # as strategies that can be formed by swapping adjacent tests
    def local_step_adjacent_swaps(self):
        best_swap = None
        best_swap_improvement = mp.eps

        for i in range(self.n - 1):
            improvement = self.gain_from_swap(i, i + 1)
            if improvement > best_swap_improvement:
                best_swap_improvement = improvement
                best_swap = (i, i + 1)
        
        if best_swap is not None:
            self.update_tables_after_swap(*best_swap)
            self.swap_tests(*best_swap)
            self.cost -= best_swap_improvement
            return True

        return False

    # Swaps tests i and j in our current strategy
    def swap_tests(self, i, j):
        self.strategy[i], self.strategy[j] = self.strategy[j], self.strategy[i]
    
    # Initializes the DP tables used by local search
    def init_local_search_tables(self):
        # pr_only[c][i] = Pr[only seen color c just after rolling (i + 1)th die]
        self.pr_only = np.ndarray(shape=(self.d, self.n), dtype=NAE_float)

        for c in range(self.d):
            self.pr_only[c][0] = self.distribution[self.strategy[0]][c]
        
        for turn in range(1, self.n):
            for c in range(self.d):
                # Pr[we only had c on the last test] * Pr[we get c on this test]
                self.pr_only[c][turn] = self.pr_only[c][turn - 1] * self.distribution[self.strategy[turn]][c]
        
        self.sum_from_to = np.zeros(shape=(self.d, self.n, self.n), dtype=NAE_float)
        self.update_partial_sums()
    
    # Updates the partial sum tables after finalizing a local search step
    def update_partial_sums(self):
        for c in range(self.d):
            for i in range(self.n):                
                self.sum_from_to[c][i][i] = self.pr_only[c][i]

                for j in range(i + 1, self.n):
                    self.sum_from_to[c][i][j] = self.sum_from_to[c][i][j - 1] + self.pr_only[c][j]
    
    # Returns the gain in expected cost from swapping test i with j
    # i.e., E[cost(old)] - E[cost(new)]
    def gain_from_swap(self, i, j):
        gain = NAE_float(0)
        for c in range(self.d):
            change_factor = self.distribution[self.strategy[j]][c] / self.distribution[self.strategy[i]][c]
            # sum_from_to has inclusive indices, so use j - 1
            gain += self.sum_from_to[c][i][j - 1] * (1 - change_factor) # = sum[i][j] - sum[i][j] * change_factor
        
        return gain
    
    # Updates the DP tables after finalizing a local search step
    def update_tables_after_swap(self, i, j):
        for c in range(self.d):
            change_factor = self.distribution[self.strategy[j]][c] / self.distribution[self.strategy[i]][c]
            for k in range(i, j):
                self.pr_only[c][k] *= change_factor
        
        self.update_partial_sums()
    
    # Performs local search until hitting a local minimum
    def local_search(self):
        self.init_local_search_tables()
        step_count = 0
        while self.local_step():
            step_count += 1
            if step_count > 10_000:
                print(self.cost)
                print(f"Performed {step_count} steps.")
        
        return step_count

    # Shuffles to a random strategy and starts the local search
    def shuffle_local_search(self):
        self.shuffle_strat()
        step_count = self.local_search()

        return step_count

ITER_COUNT = 100
def avg_local_steps(n):
    total_step_count = 0
    for _ in range(ITER_COUNT):
        nae = NAE(n, 3)
        total_step_count += nae.shuffle_local_search()
    
    return total_step_count / ITER_COUNT

def check_for_local_minima(n):
    nae = NAE(n, 3)
    for _ in range(10):
        nae.shuffle_local_search()
        print("---")

if __name__ == "__main__":
    nrange = range(10, 51)
    tick_range = nrange[::10]
    xpoints = list(nrange)
    ypoints = []
    for n in nrange:
        ypoints.append(avg_local_steps(n))
        print(f"Completed n = {n}")

    plt.plot(xpoints, ypoints, 'r-o')
    plt.ylabel(f"Local Search Iterations, {mp.dps} Decimal Places")
    plt.xlabel("n")
    plt.xticks(tick_range) # Labels on the x-axis should be integers

    plt.show()