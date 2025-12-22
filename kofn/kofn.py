# @file     kofn.py
# @author   Evan Brody
# @brief    Testbed for nonadaptive k-of-n evaluation.

import numpy as np
import itertools as it
import sys

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

    def expected_cost_printing(self, strategy):
        # ones_count[i] stores Pr[have i ones]
        ones_count = np.zeros(shape=(self.n + 1,), dtype=float)
        ones_count[0] = 1.0
        cost = 1.0 # We always flip the first coin

        one_indices = np.array([ i for i in range(self.k) ])
        zero_indices = np.array([ i for i in range(self.n - self.k + 1) ])

        for step, j in enumerate(strategy):
            step += 1 # Correct for 0-indexing

            [ print(i, end='\t') for i in one_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step] ]; print()
            print('============[ end of 1 ]============')
            [ print(i, end='\t') for i in zero_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step][::-1] ]; print()
            print('============[ end of 0 ]============')

            # Move probability mass forward according to the chosen coin
            for l in range(self.n, -1, -1):
                ones_count[l] -= ones_count[l] * self.p[j]
                if l > 0: ones_count[l] += ones_count[l - 1] * self.p[j]

            # Check which realizations aren't finished
            for num_ones in range(step):
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
        self.expected_cost_printing(kofn.OPT)
        print()
        print(kofn.EOPT)
        print(np.matrix.round(kofn.p, 2))

def array_non_decreasing(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))

def array_non_increasing(a):
    return all(a[i] >= a[i + 1] for i in range(len(a) - 1))

def array_is_sorted(a):
    return array_non_decreasing(a) or array_non_increasing(a)

K = 3
N = 5
K_BAR = N - K + 1
if __name__ == '__main__':
    threshold = min(K, K_BAR)
    one_start = set([ i for i in range(N - 1, N - K - 1, -1) ])
    zero_start = set([ i for i in range(N - K + 1) ])

    for iteration in range(1_000_000):
        kofn = KOFN(K, N)
        kofn.brute_force_OPT()
        one_starter = set(kofn.OPT[:K])
        zero_starter = set(kofn.OPT[:K_BAR])
        
        sorted_start = False
        if one_starter == one_start:
            sorted_start = True
            if not array_non_increasing(kofn.OPT[K:]):
                kofn.print_OPT()
                sys.exit(0)
        
        if zero_starter == zero_start:
            sorted_start = True
            if not array_non_decreasing(kofn.OPT[K_BAR:]):
                kofn.print_OPT()
                sys.exit(0)
        
        if not sorted_start:
            kofn.print_OPT()
            sys.exit(0)

        print(f'===============[{iteration}]===============')
