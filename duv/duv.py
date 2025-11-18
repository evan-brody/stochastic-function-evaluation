# @file     duv.py
# @author   Evan Brody
# @brief    Testbed for d-ary Unanimous Vote

import numpy as np
import itertools as it
import functools as ft
import matplotlib.pyplot as plt
import copy
import sys
import math

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
    
    def good_distribution(self):
        return not (
            all( c[0] <= 0.5 for c in self.distribution )
            or all( c[0] >= 0.5 for c in self.distribution )
        )
    
    def print_distribution(self):
        for die in self.distribution:
            for side in die:
                print(round(side, 4), end=' ')
            print()

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
    
    def generate_alt_greedy(self):
        if self.d != 2:
            raise Exception("alt greedy with d != 2")
        
        p = [ c[0] for c in self.distribution ]

        self.greedy_terms = np.zeros(shape=(2, self.n - 1))

        if all( pj >= 0.5 for pj in p ):
            self.alt_greedy = np.argsort(p) # increasing heads
            self.alt_greedy_cost = self.expected_cost(self.alt_greedy)

            for k in range(self.n):
                self.greedy_terms[0][k] = self.greedy_terms[0][k-1] * self.distribution[self.alt_greedy[k]][0]
                self.greedy_terms[1][k] = self.greedy_terms[1][k-1] * self.distribution[self.alt_greedy[k]][1]

            return
        elif all( pj <= 0.5 for pj in p ): # decreasing heads
            self.alt_greedy = np.argsort(p)[::-1]
            self.alt_greedy_cost = self.expected_cost(self.alt_greedy)

            for k in range(self.n):
                self.greedy_terms[0][k] = self.greedy_terms[0][k-1] * self.distribution[self.alt_greedy[k]][0]
                self.greedy_terms[1][k] = self.greedy_terms[1][k-1] * self.distribution[self.alt_greedy[k]][1]
            
            return
        
        self.alt_greedy = np.empty(shape=(self.n,), dtype=int)
        sorted_indexes = np.argsort(p)[::-1]
        bias = np.full(shape=(2,), dtype=float, fill_value=1.0)
        headsest_available = 0
        tailsest_available = self.n - 1

        choice = sorted_indexes[headsest_available]
        headsest_available += 1

        self.alt_greedy[0] = choice
        bias[0] *= p[choice]
        bias[1] *= 1.0 - p[choice]

        for k in range(1, self.n):
            if bias[0] <= bias[1]: # unbiased, 0-biased
                choice = sorted_indexes[headsest_available]
                headsest_available += 1
            else:
                choice = sorted_indexes[tailsest_available]
                tailsest_available -= 1
            
            self.alt_greedy[k] = choice
            bias[0] *= p[choice]
            bias[1] *= 1.0 - p[choice]

            self.greedy_terms[0][k - 1] = bias[0]
            self.greedy_terms[1][k - 1] = bias[1]

        self.alt_greedy_cost = self.expected_cost(self.alt_greedy)

    def print_alt_greedy(self):
        print("Alt Greedy:", [ int(j) for j in self.alt_greedy ])
        print("E[Alt Greedy]:", self.alt_greedy_cost)

        print("Greedy terms:")
        for heads_term in self.greedy_terms[0]:
            print(round(heads_term, 3), end=' ')
        print()

        for tails_term in self.greedy_terms[1]:
            print(round(tails_term, 3), end=' ')
        print()

        for c in range(self.d):
            for j in self.alt_greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()
    
    # Finds the expected cost of the optimal adaptive strategy, and the terms in its sum
    # Only works for d = 2, will update at some point
    def adapt_OPT_ecost(self):
        self.AOPT_terms = np.empty(shape=(2,self.n - 1), dtype=float)
        self.AOPT = float('inf')

        chosen_first_coin = None
        for first_test in range(self.n):
            p = [ c[0] for c in self.distribution ]
            del p[first_test]
            p.sort()

            these_terms = np.zeros(shape=(2,self.n - 1), dtype=float)

            heads_term = self.distribution[first_test][0] * p[0]
            tails_term = (1.0 - self.distribution[first_test][0]) * (1.0 - p[self.n - 2])

            these_terms[0][0] = heads_term
            these_terms[1][0] = tails_term

            for j in range(1, self.n - 1):
                heads_term *= p[j]
                tails_term *= 1.0 - p[self.n - 2 - j]

                these_terms[0][j] = heads_term
                these_terms[1][j] = tails_term
            
            this_cost = sum(these_terms[0]) + sum(these_terms[1])
            if this_cost < self.AOPT:
                chosen_first_coin = first_test
                self.AOPT = this_cost
                self.AOPT_terms = copy.deepcopy(these_terms)
        
        self.AOPT += 2.0

    def print_AOPT(self):
        print(f"E[AOPT]: {self.AOPT}")
        print(f"Terms:")
        for head_term in self.AOPT_terms[0]:
            print(round(head_term, 3), end=' ')
        print()
        for tails_term in self.AOPT_terms[1]:
            print(round(tails_term, 3), end=' ')
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
            new_die = self.normalize(new_die)

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

    def similarity(self):
        score = 0.0
        for die_one, die_two in it.combinations(self.distribution, 2):
            length_one = math.sqrt(sum([ p ** 2 for p in die_one ]))
            length_two = math.sqrt(sum([ p ** 2 for p in die_two ]))

            score += sum([ die_one[c] * die_two[c] for c in range(self.d) ]) / (length_one * length_two)
        
        return score

    def diff(self):
        return sum([
            self.greedy_terms[0][j] + self.greedy_terms[1][j]
            - self.AOPT_terms[0][j] - self.AOPT_terms[1][j]
            for j in range(1, self.n - 1)
        ])


GENERATION_SIZE = 10_000
GENERATION_COUNT = 1000
DN = (2, 10)
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_similarity = -1
    max_diff_instance = None
    try:
        for _ in range(100_000):
            duv = DUV(*DN)
            duv.init_distribution()

            if not duv.good_distribution():
                continue

            duv.adapt_OPT_ecost()
            duv.generate_alt_greedy()

            diff = duv.diff()
            if diff > max_diff:
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

                if not duv.good_distribution():
                    continue

                duv.adapt_OPT_ecost()
                duv.generate_alt_greedy()

                diff = duv.diff()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(duv)

                if i % 1000 == 0:
                    print(f"-------------[gen {_}, {i} -> {round(max_diff,5)}]-------------")
                
                i += 1

        print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_AOPT(); print()
        max_diff_instance.print_alt_greedy()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_AOPT(); print()
        max_diff_instance.print_alt_greedy()