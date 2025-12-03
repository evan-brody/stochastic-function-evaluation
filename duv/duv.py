# @file     duv.py
# @author   Evan Brody
# @brief    Testbed for d-ary Unanimous Vote

import numpy as np
import itertools as it
import functools as ft
import copy
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

    def generate_greedy(self):
        biases = np.ones(shape=(self.d,), dtype=float)
        available = [True] * self.n
        self.greedy = np.empty(shape=(self.n,), dtype=int)
        self.greedy_cost = 1.0
        self.greedy_terms = np.empty(shape=(self.d, self.n - 1), dtype=float)

        k = 0

        # fixed first test
        # for c in range(self.d):
        #     biases[c] *= self.distribution[self.AOPT_first_die][c]
        #     self.greedy_terms[c][0] = biases[c]
        # self.greedy[0] = self.AOPT_first_die
        # self.greedy_cost += sum(biases)
        # k += 1
        # available[self.AOPT_first_die] = False

        while k < self.n - 1:
            if np.all(biases == biases[0]): # unbiased
                # find best pair
                best_pair = (None, None)
                min_score = float('inf')
                unordered_pairs = list(it.combinations(range(self.n), 2))
                for i, j in unordered_pairs:
                    if not (available[i] and available[j]):
                        continue

                    this_pair_score = np.dot(self.distribution[i], self.distribution[j])
                    if this_pair_score < min_score:
                        min_score = this_pair_score
                        best_pair = (i, j)
                
                # update biases
                for c in range(self.d):
                    biases[c] *= self.distribution[best_pair[0]][c]
                    self.greedy_terms[c][k] = biases[c]
                self.greedy[k] = best_pair[0]
                self.greedy_cost += sum(biases)
                k += 1

                for c in range(self.d):
                    biases[c] *= self.distribution[best_pair[1]][c]
                    self.greedy_terms[c][k] = biases[c]
                self.greedy[k] = best_pair[1]
                self.greedy_cost += sum(biases)
                k += 1

                # update availability
                available[best_pair[0]] = available[best_pair[1]] = False
            else:
                best_test = None
                min_score = float('inf')
                for j in range(self.n):
                    if not available[j]:
                        continue

                    this_test_score = np.dot(biases, self.distribution[j])
                    if this_test_score < min_score:
                        min_score = this_test_score
                        best_test = j

                # update biases
                for c in range(self.d):
                    biases[c] *= self.distribution[best_test][c]
                    self.greedy_terms[c][k] = biases[c]
                self.greedy[k] = best_test
                self.greedy_cost += sum(biases)
                k += 1

                # update availability
                available[best_test] = False
        
        for j in range(self.n):
            if available[j]:
                self.greedy[self.n - 1] = j
                break
    
    def print_greedy(self):
        print("Greedy:", [ int(j) for j in self.greedy ])
        print("E[Greedy]:", self.greedy_cost)
        
        print("Greedy terms:")
        for c in range(self.d):
            for c_term in self.greedy_terms[c]:
                print(round(c_term, 3), end='\t')
            print()
        
        print()

        for c in range(self.d):
            for j in self.greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    # Finds the expected cost of the optimal adaptive strategy, and the terms in its sum
    def adapt_OPT_ecost(self):
        self.AOPT_terms = np.empty(shape=(self.d,self.n - 1), dtype=float)
        self.AOPT = float('inf')

        sorted_by = np.empty(shape=(self.d, self.n), dtype=int)
        for c in range(self.d):
            sorted_by[c] = np.argsort(self.distribution[:,c])

        self.AOPT_first_die = None
        for first_die in range(self.n):
            color_terms = np.empty(shape=(self.d, self.n - 1), dtype=float)
            for c in range(self.d):
                color_terms[c][0] = self.distribution[first_die][c]
                sorted_by_index = 0
                color_terms_index = 1
                while color_terms_index < self.n - 1:
                    if sorted_by[c][sorted_by_index] == first_die:
                        sorted_by_index += 1
                        continue
                    
                    new_factor = self.distribution[sorted_by[c][sorted_by_index]][c]
                    color_terms[c][color_terms_index] = color_terms[c][color_terms_index - 1] * new_factor
                    sorted_by_index += 1
                    color_terms_index += 1
            
            this_cost = sum([ sum(color_terms[c]) for c in range(self.d) ])
            if this_cost < self.AOPT:
                self.AOPT_first_die = first_die
                self.AOPT = this_cost
                self.AOPT_terms = copy.deepcopy(color_terms)
        
        self.AOPT += 1.0

    def print_AOPT(self):
        print(f"E[AOPT]: {self.AOPT}")
        print(f"Terms:")
        for c in range(self.d):
            for c_term in self.AOPT_terms[c]:
                print(round(c_term, 3), end='\t')
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

    def similarity(self):
        score = 0.0
        for die_one, die_two in it.combinations(self.distribution, 2):
            length_one = math.sqrt(sum([ p ** 2 for p in die_one ]))
            length_two = math.sqrt(sum([ p ** 2 for p in die_two ]))

            score += sum([ die_one[c] * die_two[c] for c in range(self.d) ]) / (length_one * length_two)
        
        return score

    def diff(self):
        sum_to_bound = 0.0
        for k in range(1, self.n - 1):
            if sum(self.greedy_terms[:,k]) > sum(self.greedy_terms[:,k-1]):
                sum_to_bound += sum(self.greedy_terms[:,k])
        
        return sum_to_bound - self.AOPT

GENERATION_SIZE = 1000
GENERATION_COUNT = 100_000
DN = (3, 8)
CAP = sum([ 0.5 ** n for n in range(2, DN[1]) ])
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_similarity = -1
    max_diff_instance = None
    try:
        for _ in range(100_000):
            duv = DUV(*DN)
            duv.init_distribution()

            duv.adapt_OPT_ecost()
            duv.generate_greedy()

            diff = duv.diff()
            if diff > max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(duv)

            if i % 1000 == 0:
                print(f"-------------[{i} -> {round(max_diff, 5)}]------------- ≤ {CAP}")
            
            i += 1
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                duv = DUV(*DN)
                duv.init_child_distribution(current_parent.distribution)

                duv.adapt_OPT_ecost()
                duv.generate_greedy()

                diff = duv.diff()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(duv)

                if i % 1000 == 0:
                    print(f"-------------[gen {_}, {i} -> {round(max_diff, 5)}]------------- ≤ {CAP}")
                
                i += 1

        print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_AOPT(); print()
        max_diff_instance.print_greedy()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_AOPT(); print()
        max_diff_instance.print_greedy()