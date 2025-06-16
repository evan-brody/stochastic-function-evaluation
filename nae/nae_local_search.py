# @file     nae_local_search.py
# @author   Evan Brody
# @brief    Tests the local convergence rate of the NAE function.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class NAE:
    RNG = np.random.default_rng()
    PRINT_STEP = 10_000
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.init_distribution()

        # Start from a random strategy
        self.shuffle_strat()
    
    def shuffle_strat(self):
        self.strategy = self.RNG.permutation(self.n)
        self.update_expected_cost()
    
    def init_distribution(self):
        # Generates random "markers" in [0, 1]
        self.distribution = np.random.rand(self.n, self.d)
        self.distribution[:, -1] = 1
        self.distribution.sort(axis=1)

        # Calculate the space between markers, which will be the
        # probability of some outcome
        for die in self.distribution:
            for i in range(self.d - 1, 0, -1):
                die[i] -= die[i - 1]

    def update_expected_cost(self):
        self.cost = 1
        prOnlySeen = np.array([ p for p in self.distribution[self.strategy[0]] ], dtype=np.longdouble)
        for test in self.strategy[1:]:
            self.cost += sum(prOnlySeen)
            for c, p in enumerate(self.distribution[test]):
                prOnlySeen[c] *= p

    # Performs one step of local search
    # Local search is done by swapping adjacent tests
    def local_step(self):
        best_swap = None
        best_swap_improvement = 0
        for i in range(self.n - 1):
            for j in range(i + 1, i + 2):
                improvement = self.gain_from_swap(i, j)
                if improvement > best_swap_improvement:
                    best_swap = (i, j)
                    best_swap_improvement = improvement
        
        if best_swap is not None:
            # Actually perform the swap
            self.strategy[best_swap[0]], self.strategy[best_swap[1]] \
                = self.strategy[best_swap[1]], self.strategy[best_swap[0]]
            self.cost = self.cost - best_swap_improvement
            self.update_tables_after_swap(*best_swap)
            return True
        
        return False
    
    def init_local_search_tables(self):
        # pr_only[c][i] = Pr[only seen color c just afte rolling (i + 1)th die]
        self.pr_only = np.ndarray(shape=(self.d, self.n))

        for c in range(self.d):
            self.pr_only[c][0] = self.distribution[self.strategy[0]][c]
        
        for turn in range(1, self.n):
            for c in range(self.d):
                self.pr_only[c][turn] = self.pr_only[c][turn - 1] * self.distribution[self.strategy[turn]][c]
        
        self.sum_from_to = np.zeros(shape=(self.d, self.n, self.n))
        for c in range(self.d):
            for i in range(self.n):
                self.sum_from_to[c][i][i] = self.pr_only[c][i]

                for j in range(i + 1, self.n):
                    self.sum_from_to[c][i][j] = self.sum_from_to[c][i][j - 1] + self.pr_only[c][j]
    
    def gain_from_swap(self, i, j):
        gain = 0
        for c in range(self.d):
            change_factor = self.distribution[self.strategy[j]][c] / self.distribution[self.strategy[i]][c]
            
            gain += self.sum_from_to[c][i][j] * (1 - change_factor) # = sum[i][j] - sum[i][j] * change_factor
        
        return gain
    
    def update_tables_after_swap(self, i, j):
        self.init_local_search_tables()
        return
        for c in range(self.d):
            change_factor = self.distribution[self.strategy[j]][c] / self.distribution[self.strategy[i]][c]
            for ii in range(i, j):
                self.pr_only[c][ii] *= change_factor
            
            for ii in range(j):
                for jj in range(max(i, ii), j):
                    self.sum_from_to[c][ii][jj] *= change_factor
    
    # Performs local search until hitting a local minimum
    def local_search(self):
        self.init_local_search_tables()
        step_count = 0
        while self.local_step():
            step_count += 1
        
        return step_count

    def shuffle_local_search(self):
        self.shuffle_strat()
        self.update_expected_cost()
        step_count = self.local_search()

        # print(f"Minimum cost: {self.cost}")

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
    # check_for_local_minima(10)

    nrange = range(10, 51)
    tick_range = [ n for n in nrange if n % 10 == 0 ]
    xpoints = list(nrange)
    ypoints = []
    for n in nrange:
        ypoints.append(avg_local_steps(n))
        print(f"Completed n = {n}")

    plt.plot(xpoints, ypoints, 'r-o')
    plt.ylabel("Local Search Iterations")
    plt.xlabel("n")
    plt.xticks(tick_range) # Labels on the x-axis should be integers

    plt.show()