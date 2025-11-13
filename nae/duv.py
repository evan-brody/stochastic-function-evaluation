# @file     duv.py
# @author   Evan Brody
# @brief    Testbed for d-ary Unanimous Vote

import numpy as np
import itertools as it
import functools as ft
import matplotlib.pyplot as plt
import copy
import sys

class DUV:
    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.distribution = np.empty(shape=(self.n, self.d), dtype=float)
        # self.init_distribution()

    # Generates a distribution over the tests and colors
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
    
    def expected_cost(self, strategy):
        cost = 1
        pr_only_seen = np.array([ p for p in self.distribution[strategy[0]] ])
        for test in strategy[1:]:
            cost += sum(pr_only_seen)
            for c, p in enumerate(self.distribution[test]):
                pr_only_seen[c] *= p
        
        return cost
    
    def generate_OPT(self):
        NUM_SUBSETS = 2 ** self.n

        opt_perms = np.ndarray(shape=(NUM_SUBSETS, self.n), dtype=int)
        pr_all_c = np.ndarray(shape=(NUM_SUBSETS, self.d), dtype=np.float64)
        opt_perms_ecost = np.ndarray(shape=(NUM_SUBSETS), dtype=np.float64)

        singletons = np.array([ 1 << j for j in range(self.n) ])
        for j, singleton in enumerate(singletons):
            opt_perms[singleton][0] = j
            pr_all_c[singleton][:] = 1
            opt_perms_ecost[singleton] = 1

        # Find best permutation for each subset
        for subset_size in range(2, self.n + 1):
            for subset in it.combinations(singletons, subset_size):
                subset = ft.reduce(lambda a, b: a | b, subset)

                # Find minimum prefix
                best_end_test = None
                best_end_test_bit = None
                cost_with_best_end_test = float('inf')

                # Pulling test j out of subset S creates a prefix
                for j, singleton in enumerate(singletons):
                    # j must be in S
                    if not (singleton & subset): continue

                    # Bitwise XOR removes j from S
                    no_j = subset ^ singleton

                    # E[cost(S)] = E[cost(S\{j})] + Pr[test j]
                    # Pr[test j] = Pr[all colors in prefix are identical]       
                    cost_with_end_j = opt_perms_ecost[no_j] + sum([ 
                        pr_all_c[no_j][c] * self.distribution[opt_perms[no_j][subset_size - 2]][c] for c in range(self.d)
                    ])
                    if cost_with_end_j < cost_with_best_end_test:
                        cost_with_best_end_test = cost_with_end_j
                        best_end_test = j
                        best_end_test_bit = singleton
                
                opt_perms[subset] = copy.deepcopy(
                    opt_perms[subset ^ best_end_test_bit]
                )
                opt_perms[subset][subset_size - 1] = best_end_test
                opt_perms_ecost[subset] = cost_with_best_end_test

                for c in range(self.d):
                    pr_all_c[subset][c] = pr_all_c[subset ^ best_end_test_bit][c] * \
                        self.distribution[opt_perms[subset][subset_size - 2]][c]
        
        # Bit representation of [n]
        ALL_TESTS = NUM_SUBSETS - 1

        self.OPT = copy.deepcopy(opt_perms[ALL_TESTS])
        self.EOPT = opt_perms_ecost[ALL_TESTS]
    
    def print_OPT(self):
        print("OPT:", [ int(j) for j in self.OPT ])
        print("E[OPT]:", self.EOPT)
        for c in range(self.d):
            for j in self.OPT:
                print(round(self.distribution[j][c], 3), end='\t')
            print()
    
    def brute_force_OPT(self):
        cost_to_beat = self.n
        for strategy in it.permutations(list(range(self.n))):
            this_strat_ecost = self.expected_cost(strategy)
            if this_strat_ecost < cost_to_beat:
                cost_to_beat = this_strat_ecost
                self.OPT = copy.deepcopy(strategy)
        
        self.EOPT = cost_to_beat

    def generate_greedy_with_first_test(self, first):
        used = np.array([False] * self.n)
        strategy = np.empty(shape=(self.n,), dtype=int)

        strategy[0] = first
        used[first] = True

        current_prs = np.array([
            self.distribution[first][c] for c in range(self.d)
        ])

        for k in range(1, self.n):
            # print(current_prs)
            best_score = self.n
            best_test = None
            for j in range(self.n):
                if used[j]: continue

                this_test_score = 0
                for c in range(self.d):
                    this_test_score += current_prs[c] * self.distribution[j][c]
                
                if this_test_score <= best_score:
                    best_score = this_test_score
                    best_test = j
            
            strategy[k] = best_test
            used[best_test] = True
            for c in range(self.d):
                current_prs[c] *= self.distribution[best_test][c]
        
        return strategy
    
    def generate_greedy(self):
        self.greedy = np.empty(shape=(self.n,), dtype=int)
        self.greedy_cost = self.n
        
        for j in range(self.n):
            starts_with_j = self.generate_greedy_with_first_test(j)
            starts_with_j_cost = self.expected_cost(starts_with_j)

            if starts_with_j_cost <= self.greedy_cost:
                self.greedy_cost = starts_with_j_cost
                self.greedy = copy.deepcopy(starts_with_j)
    
    def print_greedy(self):
        print("Greedy:", [ int(j) for j in self.greedy ])
        print("E[Greedy]:", self.greedy_cost)
        for c in range(self.d):
            for j in self.greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    def generate_double_greedy_with_first_test(self, first):
        used = np.array([False] * self.n)
        strategy = np.empty(shape=(self.n,), dtype=int)

        strategy[0] = first
        used[first] = True

        current_prs = np.array([ p for p in self.distribution[first] ])

        for k in range(1, self.n):
            if k == self.n - 1:
                for j in it.filterfalse(lambda x: used[x], range(self.n)):
                    strategy[-1] = j
            else:
                best_score = self.n
                best_pair = (None, None)
                for i, j in it.product(range(self.n), repeat=2):
                    if i == j or used[i] or used[j]: continue

                    this_pair_score = 0
                    for c in range(self.d):
                        this_pair_score += current_prs[c] * self.distribution[i][c]
                        this_pair_score += current_prs[c] * self.distribution[i][c] * self.distribution[j][c]
                    
                    if this_pair_score < best_score:
                        best_score = this_pair_score
                        best_pair = (i, j)
                
                strategy[k] = best_pair[0]
                used[best_pair[0]] = True
                for c in range(self.d):
                    current_prs[c] *= self.distribution[best_pair[0]][c]
        
        return strategy
    
    def generate_double_greedy(self):
        self.double_greedy = np.empty(shape=(self.n,), dtype=int)
        self.double_greedy_cost = self.n
        
        for j in range(self.n):
            starts_with_j = self.generate_double_greedy_with_first_test(j)
            starts_with_j_cost = self.expected_cost(starts_with_j)

            if starts_with_j_cost < self.double_greedy_cost:
                self.double_greedy_cost = starts_with_j_cost
                self.double_greedy = copy.deepcopy(starts_with_j)

    def print_double_greedy(self):
        print("DGreedy:", [ int(j) for j in self.double_greedy ])
        print("E[DGreedy]:", self.double_greedy_cost)
        for c in range(self.d):
            for j in self.double_greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    def generate_middle_greedy_with_bookends(self, first, last):
        used = np.array([False] * self.n)
        strategy = np.empty(shape=(self.n,), dtype=int)

        strategy[0] = first
        strategy[-1] = last
        used[first] = used[last] = True

        current_prs = np.array([ p for p in self.distribution[first] ])

        for k in range(1, self.n - 1):
            best_score = self.n
            best_test = None
            for j in range(self.n):
                if used[j]: continue

                this_test_score = 0
                for c in range(self.d):
                    this_test_score += current_prs[c] * self.distribution[j][c]
                
                if this_test_score < best_score:
                    best_score = this_test_score
                    best_test = j
            
            strategy[k] = best_test
            used[best_test] = True
            for c in range(self.d):
                current_prs[c] *= self.distribution[best_test][c]
        
        return strategy

    def generate_middle_greedy(self):
        self.middle_greedy = np.empty(shape=(self.n,), dtype=int)
        self.middle_greedy_cost = self.n
        
        for i, j in it.product(range(self.n), repeat=2):
            if i == j: continue
            greedy_ij = self.generate_middle_greedy_with_bookends(i, j)
            greedy_ij_cost = self.expected_cost(greedy_ij)

            if greedy_ij_cost < self.middle_greedy_cost:
                self.middle_greedy_cost = greedy_ij_cost
                self.middle_greedy = copy.deepcopy(greedy_ij)
    
    def print_middle_greedy(self):
        print("MGreedy:", [ int(j) for j in self.middle_greedy ])
        print("E[MGreedy]:", self.middle_greedy_cost)
        for c in range(self.d):
            for j in self.middle_greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    def generate_simple_greedy(self):
        self.simple_greedy = np.empty(shape=(self.n,), dtype=int)

        used = np.array([False] * self.n)

        context = np.ones(shape=(self.d,), dtype=float)

        k = 0
        while k < self.n:
            min_score = self.n
            best_test = 0

            for j in range(self.n):
                if used[j]: continue

                this_test_score = 0
                for c in range(self.d):
                    this_test_score += context[c] * self.distribution[j][c]
                
                if this_test_score < min_score:
                    min_score = this_test_score
                    best_test = j
            
            self.simple_greedy[k] = best_test
            used[best_test] = True

            for c in range(self.d):
                context[c] *= self.distribution[best_test][c]

            k += 1
        
        self.simple_greedy_cost = self.expected_cost(self.simple_greedy)

    def print_simple_greedy(self):
        print("SGreedy:", [ int(j) for j in self.simple_greedy ])
        print("E[SGreedy]:", self.simple_greedy_cost)
        for c in range(self.d):
            for j in self.simple_greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    def get_scale_vector(self):
        scale_vector = np.ones(shape=(self.d,), dtype=float)

        for c in range(self.d):
            scale_vector[c] += np.random.normal(scale=0.01)

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
        for j in range(self.n):
            new_die = copy.deepcopy(parent_distribution[j])
            scale_vector = self.get_scale_vector()
            for c in range(self.d):
                new_die[c] *= scale_vector[c]
            
            new_die = self.clamp(new_die)
            # new_die = self.normalize(new_die)

            self.distribution[j] = copy.deepcopy(new_die)
        
        return self.distribution
    
    def OPT_non_greedy(self):
        available = np.array([True] * self.n)
        self.OPT_non_greedy_indexes = []
        
        bias = np.array([1.0] * self.d)
        for k in range(self.n):
            greedy_choice = None
            for j in range(self.n):
                if available[j]:
                    greedy_choice = j
                    break
            min_score = float('inf')

            for j in [ j for j in range(self.n) if available[j] ]:
                this_test_score = 0
                for c in range(self.d):
                    this_test_score += bias[c] * self.distribution[j][c]
                
                if this_test_score < min_score:
                    min_score = this_test_score
                    greedy_choice = j

            if greedy_choice != self.OPT[k] and k != 0:
                self.OPT_non_greedy_indexes.append((k, greedy_choice))
            
            available[self.OPT[k]] = False
            for c in range(self.d):
                bias[c] *= self.distribution[self.OPT[k]][c]
        
        self.OPT_non_greedy_count = len(self.OPT_non_greedy_indexes)
    
    def plot_dice(self):
        if self.d != 3: return

        fig, ax = plt.subplots()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        for die in self.distribution: 
            ax.plot(die[0], die[1], marker='o')
        for k in range(self.n - 1):
            die = self.distribution[self.OPT[k]]
            next_die = self.distribution[self.OPT[k + 1]]
            ax.annotate(
                "",
                xytext=(die[0], die[1]),
                xy=(next_die[0], next_die[1]),
                arrowprops=dict(arrowstyle="->")
            )
        
        plt.show()


GENERATION_SIZE = 100_000
GENERATION_COUNT = 10
DN = (3, 8)
if __name__ == '__main__':
    i = 1
    max_diff = -1
    max_diff_instance = None
    try:
        for _ in range(1_000_000):
            duv = DUV(*DN)
            duv.init_distribution()

            duv.generate_OPT()
            duv.OPT_non_greedy()

            diff = duv.OPT_non_greedy_count
            if diff >= max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(duv)

            if i % 1000 == 0:
                print(f"-------------[{i} -> {round(max_diff,5)}]-------------")
            
            i += 1
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                duv = DUV(*DN)
                duv.init_child_distribution(current_parent.distribution)

                duv.generate_OPT()
                duv.OPT_non_greedy()

                diff = duv.OPT_non_greedy_count
                if diff >= max_diff: # should remove >= when not testing number of non-greedy choices
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(duv)

                if i % 1000 == 0:
                    print(f"-------------[gen {_}, {i} -> {round(max_diff,5)}]-------------")
                
                i += 1

        print()
        print(max_diff_instance.distribution)
        print(f"Count: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        print(f"Indexes: {max_diff_instance.OPT_non_greedy_indexes}"); print()
        max_diff_instance.generate_greedy()
        max_diff_instance.print_greedy()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        print(max_diff_instance.distribution)
        print(f"Count: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        print(f"Indexes: {max_diff_instance.OPT_non_greedy_indexes}"); print()
        max_diff_instance.generate_greedy()
        max_diff_instance.print_greedy()
        max_diff_instance.plot_dice()