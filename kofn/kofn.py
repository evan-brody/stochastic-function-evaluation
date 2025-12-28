# @file     kofn.py
# @author   Evan Brody
# @brief    Testbed for nonadaptive k-of-n evaluation.

import numpy as np
import copy
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

        # self.init_distribution()
    
    def init_distribution(self):
        self.p = np.random.rand(self.n)
        self.p.sort()
    
    def get_scale_vector(self):
        scale_vector = np.ones(shape=(self.n,), dtype=float)

        for j in range(self.n):
            scale_vector[j] += np.random.normal(scale=0.01)

        return scale_vector
    
    def normalize(self, vector):
        sv = sum(vector)
        vector[:] /= sv

        return vector
    
    def clamp(self, vector):
        for j in range(len(vector)):
            vector[j] = min(1, vector[j])
        
        return vector
    
    def init_child_distribution(self, parent_distribution):
        new_coins = copy.deepcopy(parent_distribution)
        scale_vector = self.get_scale_vector()
        for j in range(self.n):
            new_coins[j] *= scale_vector[j]
        
        new_coins = self.clamp(new_coins)
        self.p = sorted(new_coins)
        
    def nudge_k(self):
        val = np.random.rand()
        if val < 0.05:
            self.k -= 1
        elif val > 0.95:
            self.k += 1
    
    def expected_cost(self, strategy):
        # ones_count[i] stores Pr[have i ones]
        ones_count = np.zeros(shape=(self.n + 1,), dtype=float)
        ones_count[0] = 1.0
        cost = 1.0 # We always flip the first coin

        # Loop ends early because we shouldn't add in the last term
        for step, j in enumerate(strategy[:-1]):
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

        one_indices = np.array([ i for i in range(self.k) ])
        zero_indices = np.array([ i for i in range(self.n - self.k + 1) ])

        [ print(i, end='\t') for i in one_indices ]; print()
        [ print(round(f, 2), end='\t') for f in ones_count[:1] ]; print()
        print('============[ end of 1 ]============')
        [ print(i, end='\t') for i in zero_indices ]; print()
        [ print(round(f, 2), end='\t') for f in ones_count[:1][::-1] ]; print()
        print('============[ end of 0 ]============'); print()

        for step, j in enumerate(strategy):
            step += 1 # Correct for 0-indexing

            # Move probability mass forward according to the chosen coin
            for l in range(self.n, -1, -1):
                ones_count[l] -= ones_count[l] * self.p[j]
                if l > 0: ones_count[l] += ones_count[l - 1] * self.p[j]

            [ print(i, end='\t') for i in one_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step + 1] ]; print()
            print('============[ end of 1 ]============')
            [ print(i, end='\t') for i in zero_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step + 1][::-1] ]; print()
            print('============[ end of 0 ]============'); print()

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
        print(self.OPT)
        print(tuple([ float(round(self.p[j], 2)) for j in self.OPT ]))
        print(self.EOPT)
        print()

    def pm_forward_to_goal(self):
        pass
    
    def generate_one_shot(self):
        best_strategy = None
        to_beat = float('inf')
        for nondecreasing in (True, False):
            strategy = sorted(list(range(self.n)), reverse=(not nondecreasing))
            if self.k < self.k_bar and nondecreasing or self.k > self.k_bar and not nondecreasing:
                crossover = max(self.k, self.k_bar)
                strategy[:crossover] = sorted(strategy[:crossover], reverse=nondecreasing)
            
            this_cost = self.expected_cost(strategy)
            if this_cost < to_beat:
                to_beat = this_cost
                best_strategy = strategy
            
        
        self.one_shot = copy.deepcopy(best_strategy)
        self.one_shot_cost = to_beat
    
    def print_one_shot(self):
        print(self.unordered_threshold_visual)
        print(tuple(self.one_shot))
        print(tuple([ float(round(self.p[j], 2)) for j in self.one_shot ]))
        print(self.one_shot_cost); print()
    
    def diff(self):
        return self.one_shot_cost - self.EOPT
        

def array_non_decreasing(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))

def array_non_increasing(a):
    return all(a[i] >= a[i + 1] for i in range(len(a) - 1))

def array_is_sorted(a):
    return array_non_decreasing(a) or array_non_increasing(a)


GENERATION_SIZE = 1000
GENERATION_COUNT = 100_000
N = 6
K = 3
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_diff_instance = None

    for i in range(100_000):
        K = np.random.randint(N) + 1
        kofn = KOFN(K, N)
        kofn.init_distribution()
        kofn.brute_force_OPT()

        if not array_is_sorted(kofn.OPT[max(kofn.k, kofn.k_bar):]):
            kofn.print_OPT()
            sys.exit(0)
        
        if i % 1000 == 0:
            print(f"----------------[{i}]----------------")
    
    sys.exit(0)

    try:
        for _ in range(100_000):
            K = np.random.randint(N) + 1
            kofn = KOFN(K, N)
            kofn.init_distribution()

            kofn.brute_force_OPT()
            kofn.generate_one_shot()

            diff = kofn.diff()
            if diff > max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(kofn)

            if i % 1000 == 0:
                print(f"-------------[{i} -> {round(max_diff, 5)}]-------------")
            
            i += 1
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                kofn = KOFN(current_parent.k, N)
                kofn.init_child_distribution(current_parent.p)
                kofn.nudge_k()

                kofn.brute_force_OPT()
                kofn.generate_one_shot()

                diff = kofn.diff()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(kofn)

                if i % 1000 == 0:
                    print(f"-------------[gen {_}, K = {max_diff_instance.k} {i} -> {round(max_diff, 5)}]-------------")
                
                i += 1

        print()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        max_diff_instance.print_one_shot()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        max_diff_instance.print_one_shot()