# @file     kofn.py
# @author   Evan Brody
# @brief    Testbed for nonadaptive k-of-n evaluation.

import numpy as np
import itertools as it

class KOFN:
    def __init__(self, k, n):
        self.k = k
        self.n = n

        self.init_distribution()
    
    def init_distribution(self):
        self.p = np.random.rand(self.n)
        self.p.sort()
    
    def expected_cost(self, strategy):
        # ones_count[i] stores Pr[have i ones]
        ones_count = np.zeros(shape=(self.n + 1,), dtype=float)
        ones_count[0] = 1.0
        zeroes_count = np.zeros(shape=(self.n + 1,), dtype=float)
        zeroes_count[0] = 1.0
        cost = 1.0 # Always flip the first coin

        for step, j in enumerate(strategy):
            step += 1 # Correct for 0-indexing

            # Move probability mass forward according to the chosen coin
            for l in range(self.n, -1, -1):
                ones_count[l] -= ones_count[l] * self.p[j]
                if l > 0: ones_count[l] += ones_count[l - 1] * self.p[j]

                zeroes_count[l] -= zeroes_count[l] * (1.0 - self.p[j])
                if l > 0: zeroes_count[l] += zeroes_count[l - 1] * (1.0 - self.p[j])

            # Check which realizations aren't finished
            for l in range(step + 1):
                num_ones = l
                num_zeroes = step - l
                if num_ones < self.k and num_zeroes < self.n - self.k + 1:
                    cost += ones_count[l]

        return cost

    def brute_force_OPT(self):
        self.OPT = None
        self.EOPT = float('inf')

        # Brute-force search over every permutation
        for perm in it.permutations(list(range(self.n))):
            this_cost = self.expected_cost(perm)
            if this_cost < self.EOPT:
                self.EOPT = this_cost
                self.OPT = perm

if __name__ == '__main__':
    kofn = KOFN(5, 9)
    kofn.brute_force_OPT()
    print(kofn.OPT)
    print(kofn.EOPT)
    print(np.matrix.round(kofn.p, 2))