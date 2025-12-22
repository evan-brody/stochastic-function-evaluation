# @file     kofn.py
# @author   Evan Brody
# @brief    Testbed for nonadaptive k-of-n evaluation.

import numpy as np
import itertools as it

class KOFN:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.k_bar = self.n - self.k + 1

        self.unordered_threshold = min(K, N - K + 1)
        self.unordered_threshold_visual = [0] * N
        self.unordered_threshold_visual[min(K, N - K + 1) - 1] = 1
        self.unordered_threshold_visual = tuple(self.unordered_threshold_visual)

        self.init_distribution()
    
    def init_distribution(self):
        self.p = np.random.rand(self.n)
        self.p.sort()
    
    def expected_cost(self, strategy):
        # ones_count[i] stores Pr[have i ones]
        ones_count = np.zeros(shape=(self.n + 1,), dtype=float)
        ones_count[0] = 1.0
        cost = 1.0 # We always flip the first coin

        for step, j in enumerate(strategy):
            step += 1 # Correct for 0-indexing

            # Move probability mass forward according to the chosen coin
            for l in range(self.n, -1, -1):
                ones_count[l] -= ones_count[l] * self.p[j]
                if l > 0: ones_count[l] += ones_count[l - 1] * self.p[j]

            # Check which realizations aren't finished
            for num_ones in range(step + 1):
                num_zeroes = step - num_ones
                if num_ones < self.k and num_zeroes < self.k_bar:
                    cost += ones_count[num_ones]

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
    
    def print_OPT(self):
        print(self.unordered_threshold_visual)
        print(kofn.OPT)
        print(tuple([ float(round(kofn.p[j], 2)) for j in kofn.OPT ]))
        print()
        print(kofn.EOPT)
        print(np.matrix.round(kofn.p, 2))

K = 3
N = 7
if __name__ == '__main__':
    kofn = KOFN(K, N)
    kofn.brute_force_OPT()

    Z = [0] * N
    threshold = min(K, N - K + 1)
    Z[threshold - 1] = 1
    Z = tuple(Z)
    
    one_start = set([ i for i in range(K) ])
    zero_start = set([ i for i in range(N - 1, K - 2, -1) ])
    for iteration in range(100_000):
        kofn = KOFN(K, N)
        kofn.brute_force_OPT()
        starter = set(kofn.OPT[:threshold - 1])
        
        if not (starter.issubset(one_start) or starter.issubset(zero_start)):
            kofn.print_OPT()
            raise Exception('counterexample')
        else:
            print(f'===============[{iteration}]===============')
