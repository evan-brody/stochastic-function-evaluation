# @file     kofn.py
# @author   Evan Brody
# @brief    Testbed for nonadaptive k-of-n evaluation.

import numpy as np
import copy
import itertools as it
import sys

# Returns a vector with numbers perturbed a small amount from 1
def get_scale_vector(n):
    scale_vector = np.ones(shape=(n,), dtype=float)

    for j in range(n):
        scale_vector[j] += np.random.normal(scale=0.001)

    return scale_vector

# Returns the 1-normalized form of the given vector
def normalize(vector):
    vector[:] /= sum(vector)
    return vector

# Returns the given vector with each entry clamped to [0, 1]
def clamp(vector):
    for j in range(len(vector)):
        vector[j] = min(max(0.0, vector[j]), 1.0)
    
    return vector


def array_non_decreasing(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))

def array_non_increasing(a):
    return all(a[i] >= a[i + 1] for i in range(len(a) - 1))

def array_is_sorted(a):
    return array_non_decreasing(a) or array_non_increasing(a)


# Simulates an instance of nonadaptive k-of-n evaluation
class KOFN:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.k_bar = self.n - self.k + 1
    
    # Useful for inducing a non-sorted OPT
    def halve_distribution(self):
        self.p[:] /= 2
    
    # Initializes a product distribution over the variables
    # If parent distribution is None, then each pi is drawn from a uniform distribution on [0, 1]
    # Otherwise, initializes a distribution that deviates slightly from the parent's
    def init_distribution(self, parent_distribution=None):
        if parent_distribution is None:
            self.p = np.random.rand(self.n)
            self.p.sort()
        else:
            new_distribution = copy.deepcopy(parent_distribution)
            scale_vector = get_scale_vector(self.n)
            for j in range(self.n):
                new_distribution[j] *= scale_vector[j]
            
            new_distribution = clamp(new_distribution)
            self.p = sorted(new_distribution)
    
    # Has a small chance to decrement or increment k
    def nudge_k(self):
        val = np.random.rand()
        if val < 0.05:
            self.k -= 1
        elif val > 0.95:
            self.k += 1
    
    # Returns the expected cost of a given nonadaptive strategy
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

    # Finds the probability that f(x) = 0 and that f(x) = 1
    def find_pr_one_zero(self):
        # ones_count[i] stores Pr[have i ones]
        ones_count = np.zeros(shape=(self.n + 1,), dtype=float)
        ones_count[0] = 1.0

        for j in range(self.n):
            # Move probability mass forward
            for l in range(self.n, -1, -1):
                ones_count[l] -= ones_count[l] * self.p[j]
                if l > 0: ones_count[l] += ones_count[l - 1] * self.p[j]

        self.pr_f_one = sum(ones_count[self.k:])
        self.pr_f_zero = sum(ones_count[:self.k])

        return self.pr_f_zero, self.pr_f_one

    # Prints out a visual of the dynamic programming table used to
    # calculate the expected cost at each step
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

            print(f'Choice: {self.p[j]:.2f}')
            [ print(i, end='\t') for i in one_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step + 1] ]; print()
            print('============[ end of 1 ]============')
            [ print(i, end='\t') for i in zero_indices ]; print()
            [ print(round(f, 2), end='\t') for f in ones_count[:step + 1][::-1] ]; print()
            print('============[ end of 0 ]============'); print()
    
    # Brute force search for the optimal nonadaptive strategy
    def brute_force_OPT(self):
        self.OPT = None
        self.EOPT = float('inf')

        # Prior to this index, we can sort the coins    
        threshold = max(self.k, self.k_bar)
        starter_nonincreasing = self.k <= self.k_bar

        all_tests_set = set(range(self.n))

        for starting in it.combinations(range(self.n), threshold):
            starting = sorted(starting, reverse=starter_nonincreasing)
            remaining = all_tests_set.difference(starting)

            for ending in it.permutations(remaining):
                this_permutation = starting + list(ending)
                this_permutation_cost = self.expected_cost(this_permutation)
                if this_permutation_cost < self.EOPT:
                    self.EOPT = this_permutation_cost
                    self.OPT = this_permutation

        self.OPT = tuple(self.OPT)

    # Prints out information about OPT
    def print_OPT(self):
        print(self.OPT)
        print(tuple([ float(round(self.p[j], 2)) for j in self.OPT ]))
        print(self.EOPT); print()

    # Generates a nonadaptive strategy that starts with a sorted permutation
    # then sorts the portion before max(k, kbar), if this is advantageous
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
    
    # Prints information about the above strategy
    def print_one_shot(self):
        print(tuple(self.one_shot))
        print(tuple([ float(round(self.p[j], 2)) for j in self.one_shot ]))
        print(self.one_shot_cost); print()
    
    # Checks if a strategy always uses extremal variables
    def check_strategy_extremal(self, strategy):
        if self.k == 1 or self.k == self.n: return True

        crossover = max(self.k, self.k_bar)
        starter = set(strategy[:crossover])

        all_tests = set(range(self.n))
        not_in_starter = all_tests.difference(starter)


        upper = max(not_in_starter)
        lower = min(not_in_starter)

        # if upper - lower + 1 != len(not_in_starter):
        #     return False

        for k in range(crossover, self.n):
            if strategy[k] == upper:
                upper -= 1
            elif strategy[k] == lower:
                lower += 1
            else:
                return False
        
        return True
    
    # Checks if OPT always uses extremal variables
    # DISPROVED
    def check_OPT_extremal(self):
        res = self.check_strategy_extremal(self.OPT)
        if res: return True

        return False

    # Checks if OPT only has variables on one side of 1/2
    def OPT_unordered_biased(self):
        threshold = max(self.k, self.k_bar)
        upper = all(self.p[j] >= 0.5 for j in self.OPT[:threshold])
        lower = all(self.p[j] <= 0.5 for j in self.OPT[:threshold])

        return lower or upper

    # Tests the conjecture that the fully ordered portion (after max(k, kbar)) of OPT is sorted
    # OPEN
    def OPT_sorted_ordered(self):
        return array_is_sorted(self.OPT[max(self.k, self.k_bar):])
    
    # Generates the best sorted strategy
    # Useful when k = kbar
    def sorted_strategy(self):
        self.sorted_ascending = tuple(range(self.n))
        self.sorted_descending = tuple(list(range(self.n))[::-1])

        self.sorted_ascending_cost = self.expected_cost(self.sorted_ascending)
        self.sorted_descending_cost = self.expected_cost(self.sorted_descending)

        if self.sorted_ascending_cost <= self.sorted_descending_cost:            
            self.sorted = self.sorted_ascending
            self.sorted_cost = self.sorted_ascending_cost
        else:
            self.sorted = self.sorted_descending
            self.sorted_cost = self.sorted_descending_cost
    
    # Prints information about the above strategy
    def print_sorted(self):
        print(tuple(self.sorted))
        print(tuple([ float(round(self.p[j], 2)) for j in self.sorted ]))
        print(self.sorted_cost); print()
    
    # Used in the evolutionary algorithm in main, which seeks to maximize the return value of this function
    def diff(self):
        self.brute_force_OPT()
        self.generate_one_shot()
        return self.one_shot_cost - self.EOPT
        if self.one_shot_cost - self.EOPT < 0.001:
            return -100
        return -(self.one_shot_cost - self.EOPT - 0.001)
    
    # Prints information relevant to the evolutionary algorithm in main
    def diff_info(self):
        self.print_OPT()
        self.print_one_shot()
        print(f'Sum: {sum([ 2.0 * self.p[j] - 1.0 for j in range(self.n) ])}')


GENERATION_SIZE = 1000
GENERATION_COUNT = 1000
PRINT_PER = 1000
N = 8
K = 4

# Uses an evolutionary algorithm to optmize some value of interest
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_diff_instance = None

    try:
        # for _ in range(1_000_000):
        #     # K = np.random.randint(N) + 1
        #     kofn = KOFN(K, N)
        #     kofn.init_distribution()

        #     diff = kofn.diff()
        #     if diff > max_diff:
        #         max_diff = diff
        #         max_diff_instance = copy.deepcopy(kofn)

        #     if i % PRINT_PER == 0:
        #         print(f"-------------[K = {max_diff_instance.k}, {i} -> {round(max_diff, 5)}]-------------")
            
        #     i += 1
        
        max_diff_instance = KOFN(4, 8)
        max_diff_instance.p = [0.11, 0.31] + [0.42] * 6
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                kofn = KOFN(current_parent.k, N)
                kofn.init_distribution(current_parent.p)
                # kofn.nudge_k()

                diff = kofn.diff()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(kofn)

                if i % PRINT_PER == 0:
                    print(f"-------------[gen {_}, K = {max_diff_instance.k}, {i} -> {round(max_diff, 5)}]-------------")
                
                i += 1

        print()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.diff_info()

        print()
        for pi in max_diff_instance.p:
            print(pi)
        print()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.diff_info()

        print()
        for pi in max_diff_instance.p:
            print(pi)
        print()