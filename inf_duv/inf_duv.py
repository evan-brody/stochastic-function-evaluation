# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite dice rolling sequences

import numpy as np
import itertools as it
import functools as ft
import copy

class DUV:
    def __init__(self, d, n, k):
        self.d = d
        self.n = n
        self.k = k
        self.distribution = np.empty(shape=(self.k, self.d), dtype=float)
        # self.init_distribution()

    # Generates a distribution over the tests and colors
    def init_distribution(self):
        # Generates random "markers" in [0, 1]
        self.distribution = np.random.rand(self.k, self.d)

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
    
    def print_OPT(self):
        print("OPT:", [ int(j) for j in self.OPT ])
        print("E[OPT]:", self.EOPT)
        for c in range(self.d):
            for j in self.OPT:
                print(round(self.distribution[j][c], 3), end='\t')
            print()
    
    def brute_force_OPT(self):
        cost_to_beat = float('inf')
        for strategy in it.product(range(self.k), repeat=self.n):
            this_strat_ecost = self.expected_cost(strategy)
            if this_strat_ecost < cost_to_beat:
                cost_to_beat = this_strat_ecost
                self.OPT = copy.deepcopy(strategy)
        
        self.EOPT = cost_to_beat

    def generate_greedy_with_first_test(self, first):
        strategy = np.empty(shape=(self.n,), dtype=int)

        strategy[0] = first

        current_prs = np.array([
            self.distribution[first][c] for c in range(self.d)
        ])

        for m in range(1, self.n):
            best_score = float('inf')
            best_test = None
            for j in range(self.k):

                this_test_score = 0
                for c in range(self.d):
                    this_test_score += current_prs[c] * self.distribution[j][c]
                
                if this_test_score < best_score:
                    best_score = this_test_score
                    best_test = j
            
            strategy[m] = best_test
            for c in range(self.d):
                current_prs[c] *= self.distribution[best_test][c]
        
        return strategy
    
    def generate_greedy(self):
        self.greedy = np.empty(shape=(self.n,), dtype=int)
        self.greedy_cost = float('inf')
        
        for j in range(self.k):
            starts_with_j = self.generate_greedy_with_first_test(j)
            starts_with_j_cost = self.expected_cost(starts_with_j)

            if starts_with_j_cost < self.greedy_cost:
                self.greedy_cost = starts_with_j_cost
                self.greedy = copy.deepcopy(starts_with_j)
    
    def print_greedy(self):
        print("Greedy:", [ int(j) for j in self.greedy ])
        print("E[Greedy]:", self.greedy_cost)
        for c in range(self.d):
            for j in self.greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    # TODO: change to accomodate k < n
    # def generate_simple_greedy(self):
    #     self.simple_greedy = np.empty(shape=(self.n,), dtype=int)

    #     context = np.ones(shape=(self.d,), dtype=float)

    #     k = 0
    #     while k < self.n:
    #         min_score = self.n
    #         best_test = 0

    #         for j in range(self.n):
    #             this_test_score = 0
    #             for c in range(self.d):
    #                 this_test_score += context[c] * self.distribution[j][c]
                
    #             if this_test_score < min_score:
    #                 min_score = this_test_score
    #                 best_test = j
            
    #         self.simple_greedy[k] = best_test

    #         for c in range(self.d):
    #             context[c] *= self.distribution[best_test][c]

    #         k += 1
        
    #     self.simple_greedy_cost = self.expected_cost(self.simple_greedy)

    # def print_simple_greedy(self):
    #     print("SGreedy:", [ int(j) for j in self.simple_greedy ])
    #     print("E[SGreedy]:", self.simple_greedy_cost)
    #     for c in range(self.d):
    #         for j in self.simple_greedy:
    #             print(round(self.distribution[j][c], 3), end='\t')
    #         print()

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
            new_die = self.normalize(new_die)

            self.distribution[j] = copy.deepcopy(new_die)
        
        return self.distribution


GENERATION_SIZE = 1000
GENERATION_COUNT = 100
DNK = (3, 6, 3)
if __name__ == '__main__':
    i = 1
    max_diff = 0
    max_diff_instance = None

    for _ in range(100_000):
        duv = DUV(*DNK)
        duv.init_distribution()

        duv.brute_force_OPT()
        duv.generate_greedy()

        diff = duv.greedy_cost - duv.EOPT
        if diff > max_diff:
            max_diff = diff
            max_diff_instance = copy.deepcopy(duv)

        if i % 1000 == 0:
            print(f"-------------[{i} -> {round(max_diff,5)}]-------------")
        
        i += 1
    
    for _ in range(GENERATION_COUNT):
        current_parent = copy.deepcopy(max_diff_instance)
        for __ in range(GENERATION_SIZE):
            duv = DUV(*DNK)
            duv.init_child_distribution(current_parent.distribution)

            duv.brute_force_OPT()
            duv.generate_greedy()

            diff = duv.greedy_cost - duv.EOPT

            if diff > max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(duv)

            if i % GENERATION_SIZE == 0:
                print(f"-------------[gen {_}, {i} -> {round(max_diff,5)}]-------------")
            
            i += 1

    print(max_diff); print()
    max_diff_instance.print_OPT(); print()
    max_diff_instance.print_greedy(); print()