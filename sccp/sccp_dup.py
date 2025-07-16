# @file     sccp_dup.py
# @author   Evan Brody
# @brief    Simulates the d = n case of the Stochastic Coupon Collection Problem

import numpy as np
import itertools as it
import copy
from mpmath import mp
import functools as ft
import math

# Configure mp
mp.dps = 5     # Decimal places used by mp.mpf
mp.pretty = True # Turn pretty-printing on

SCCP_float = np.float64

class SCCP:
    def __init__(self, n):
        self.n = n
        self.init_distribution()

        # self.n = 4
        # self.distribution = np.array([[0.8, 0.2, 0.0, 0.0],
        #                                 [0.8, 0.2, 0.0, 0.0],
        #                                 [0.2, 0.0, 0.8, 0.0],
        #                                 [0.1, 0.0, 0.8, 0.1]])
    
    def init_distribution(self):
        self.distribution = np.random.rand(self.n, self.n)
        for i in range(self.n):
            self.distribution[i, :] /= sum(self.distribution[i]) # Rows should add to 1
    
    def print_distribution(self):
        print(np.matrix.round(self.distribution, 3))
    
    def ecost(self, permutation):
        NUM_STATES = 2 ** self.n

        state_vectors = np.empty(shape=(self.n, NUM_STATES), dtype=SCCP_float)
        state_vectors[0][0] = 1
        for state in range(1, NUM_STATES):
            state_vectors[0][state] = 0

        E = SCCP_float(1)
        
        for turn, die in enumerate(permutation):
            if turn == self.n - 1: break # This turn doesn't count
            state_vectors[turn + 1][0] = 0
            
            for state in range(1, NUM_STATES):
                state_vectors[turn + 1][state] = 0
                for color in range(self.n):
                    color_bit = 1 << color
                    if not (state & color_bit): continue

                    state_without_color = state ^ color_bit

                    # Add Pr[just got this color] + Pr[already had this color]
                    state_vectors[turn + 1][state] += \
                        self.distribution[die][color] * \
                        (state_vectors[turn][state] + state_vectors[turn][state_without_color])

                if int.bit_count(state) == turn + 1:
                    # We're only still going if we have exactly i colors after the ith roll
                    E += state_vectors[turn + 1][state]
        
        return E
    
    def calculate_OPT(self):
        NUM_TEST_SETS = 2 ** self.n
        NUM_COLOR_SETS = 2 ** self.n

        # A subset S is represented as a bitstring (i.e., int) where a 1 at bit j means j is in S
        # Indexing: subset (bitstring), permutation position
        optimal_permutations = np.full(shape=(NUM_TEST_SETS, self.n), fill_value=-1, dtype=int)
        optimal_costs = np.zeros(shape=(NUM_TEST_SETS), dtype=SCCP_float)

        # [S, C] = Probability of the optimal permutation of S taking on all colors in C
        pr_unique_outcomes = np.zeros(shape=(NUM_TEST_SETS, NUM_COLOR_SETS), dtype=SCCP_float)

        singletons = np.empty(self.n, dtype=int)
        for j in range(self.n):
            singletons[j] = 1 << j

        # Base cases are immediate
        for j, singleton in enumerate(singletons):
            optimal_permutations[singleton, 0] = j

            for c, c_singleton in enumerate(singletons):
                pr_unique_outcomes[singleton, c_singleton] = self.distribution[j, c]
            
            optimal_costs[singleton] = 1
        
        # Find optimal permutations for all subsets with size >= 2
        for subset_size in range(2, self.n + 1):
            for subset in it.combinations(singletons, subset_size):
                subset = ft.reduce(lambda a, b: a | b, subset)

                # Find minimum prefix
                best_end_test = None
                best_end_test_bit = None
                cost_with_best_end_test = self.n + 1

                # Pulling test j out of subset S creates a prefix
                for j, singleton in enumerate(singletons):
                    # j must be in S
                    if not (singleton & subset): continue

                    # Bitwise XOR removes j from S
                    no_j = subset ^ singleton
                    opt_cost_no_j = optimal_costs[no_j]

                    # Pr[test j] = Pr[all colors in prefix are unique]
                    # We sum over disjoint outcomes to calculate this value
                    pr_need_test_j = sum(pr_unique_outcomes[no_j])
                    
                    cost_with_end_j = opt_cost_no_j + pr_need_test_j
                    if cost_with_end_j < cost_with_best_end_test:
                        cost_with_best_end_test = cost_with_end_j
                        best_end_test = j
                        best_end_test_bit = singleton
                
                optimal_permutations[subset] = copy.deepcopy(
                    optimal_permutations[subset ^ best_end_test_bit]
                )
                optimal_permutations[subset][subset_size - 1] = best_end_test
                optimal_costs[subset] = cost_with_best_end_test

                # The next loop updates pr_unique_outcomes for our current subset S
                # Let j be the chosen test. We iterate over
                # colors that S\{j} could hold s.t. each is unique
                no_j = subset ^ best_end_test_bit
                for c_subset_no_j in it.combinations(singletons, subset_size - 1):
                    # We need the tuple that it.combinations produces to be a bitstring
                    c_subset_no_j = ft.reduce(lambda a, b: a | b, c_subset_no_j)

                    for c_bit_j_could_be in range(self.n):
                        # Need both the integer and bit representation
                        c_j_could_be = c_bit_j_could_be
                        c_bit_j_could_be = 1 << c_bit_j_could_be
                        # If we already have this color
                        if c_bit_j_could_be & c_subset_no_j: continue

                        new_unique_outcome = c_subset_no_j | c_bit_j_could_be
                        pr_new_unique_outcome = pr_unique_outcomes[no_j][c_subset_no_j] * \
                            self.distribution[best_end_test][c_j_could_be]

                        pr_unique_outcomes[subset][new_unique_outcome] += pr_new_unique_outcome
        
        ALL_TESTS = NUM_TEST_SETS - 1
        self.OPT = optimal_permutations[ALL_TESTS]
        self.EOPT = optimal_costs[ALL_TESTS]

        print(self.OPT)
        print(self.EOPT)

    # Brute-force search for the optimal permutation
    # DEPRECATED: USE CALCULATE_OPT() TO AVOID EXPONENTIAL COMPLEXITY
    # def find_OPT(self):
    #     self.OPT = None
    #     self.EOPT = SCCP_float(self.n)
    #     for permutation in it.permutations(np.array([ i for i in range(self.n) ])):
    #         this_cost = self.ecost(permutation)

    #         if this_cost <= self.EOPT:
    #             self.EOPT = this_cost
    #             self.OPT = permutation

    #     print([ int(die) for die in self.OPT ])
    #     print(self.EOPT)
        
    def ecost_color_get_one(self, color, queue, selected):
        E = SCCP_float(1)

        product = SCCP_float(1)
        for test in queue[:self.n - 1]:
            if selected[test]: continue
            product *= self.distribution[test, color]
            E += product
        
        return E

    def ecost_color_get_two(self, color, queue, selected):
        # Indexing: turn, count
        pr_have_k = np.zeros(shape=(self.n + 1, 2), dtype=SCCP_float)
        pr_have_k[0][0] = 1

        E = SCCP_float(0)
        num_tested = 0
        for test in queue:
            if selected[test]: continue

            E += pr_have_k[num_tested, 0] + pr_have_k[num_tested, 1]

            pr_have_k[num_tested + 1, 0] = pr_have_k[num_tested, 0] * (1 - self.distribution[test, color])
            pr_have_k[num_tested + 1, 1] = pr_have_k[num_tested, 0] * self.distribution[test, color] \
                + pr_have_k[num_tested, 1] * (1 - self.distribution[test, color])

            num_tested += 1
        
        return E

    def generate_greedy(self):
        self.greedy_color_pick = np.empty(self.n, int)
        self.greedy = np.empty(self.n, int)
        selected = np.array([False] * self.n)

        # Indexing: turn, count, color
        pr_have_k = np.zeros(shape=(self.n + 1, 3, self.n), dtype=SCCP_float)
        pr_have_k[0, 0, :] = 1

        queues = np.empty(shape=(self.n, self.n), dtype=int)
        for c in range(self.n):
            queues[c] = self.distribution[:, c].argsort()[::-1]

        for turn in range(self.n):
            # Select a color to focus on
            color_choice = None
            # Lower score is better
            color_choice_score = SCCP_float(float('inf'))

            for c in range(self.n):
                this_color_score = SCCP_float(0)
                
                pr_sum = pr_have_k[turn, 1, c] + pr_have_k[turn, 0, c]
                one_portion = pr_have_k[turn, 1, c] / pr_sum
                zero_portion = pr_have_k[turn, 0, c] / pr_sum
                this_color_score += self.ecost_color_get_one(c, queues[c], selected) * one_portion
                this_color_score += self.ecost_color_get_two(c, queues[c], selected) * zero_portion

                if round(one_portion + zero_portion, 7) != 1:
                    print(one_portion + zero_portion)
                    raise Exception("abd")

                if this_color_score < color_choice_score:
                    color_choice_score = this_color_score
                    color_choice = c
            
            self.greedy_color_pick[turn] = color_choice

            choice = None
            i = 0
            while i < self.n:
                if not selected[queues[color_choice, i]]:
                    choice = queues[color_choice, i]
                    break
                i += 1

            if i == self.n:
                raise Exception("Error in greedy generation.")

            # Update probabilities
            for c in range(self.n):
                pr_have_k[turn + 1, 0, c] = pr_have_k[turn, 0, c] * (1 - self.distribution[choice, c])
            
            for c in range(self.n):
                pr_have_k[turn + 1, 1, c] = pr_have_k[turn, 0, c] * self.distribution[choice, c] \
                    + pr_have_k[turn, 1, c] * (1 - self.distribution[choice, c])
            
            for c in range(self.n):
                pr_have_k[turn + 1, 2, c] = pr_have_k[turn, 1, c] * self.distribution[choice, c] \
                    + pr_have_k[turn, 2, c]

            # Insert choice and update selected
            self.greedy[turn] = choice
            selected[choice] = True
        
        self.greedy_cost = self.ecost(self.greedy)
        print(np.matrix.round(pr_have_k, 3))
    
    def generate_greedy_alt(self):
        self.greedy_alt = np.empty(self.n, int)
        self.greedy_alt_color_pick = np.empty(self.n, int)
        selected = np.array([False] * self.n)

        # Indexing: turn, count, color
        pr_have_k = np.zeros(shape=(self.n + 1, 3, self.n), dtype=SCCP_float)
        pr_have_k[0, 0, :] = 1

        queues = np.empty(shape=(self.n, self.n), dtype=int)
        for c in range(self.n):
            queues[c] = self.distribution[:, c].argsort()[::-1]

        for turn in range(self.n):
            # Select a color to focus on
            color_choice = 0
            # Lower score is better
            color_choice_score = SCCP_float(float('inf'))

            for c in range(self.n):
                this_color_score = SCCP_float(0)

                pr_sum = pr_have_k[turn, 1, c] + pr_have_k[turn, 0, c]
                one_portion = pr_have_k[turn, 1, c] / pr_sum
                zero_portion = pr_have_k[turn, 0, c] / pr_sum
                this_color_score += self.ecost_color_get_one(c, queues[c], selected) * pr_have_k[turn, 1, c]
                this_color_score += self.ecost_color_get_two(c, queues[c], selected) * pr_have_k[turn, 0, c]

                if this_color_score < color_choice_score:
                    color_choice_score = this_color_score
                    color_choice = c
            
            self.greedy_alt_color_pick[turn] = color_choice

            choice = None
            i = 0
            while i < self.n:
                if not selected[queues[color_choice, i]]:
                    choice = queues[color_choice, i]
                    break
                i += 1

            if i == self.n:
                raise Exception("Error in greedy generation.")

            # Update probabilities
            for c in range(self.n):
                pr_have_k[turn + 1, 0, c] = pr_have_k[turn, 0, c] * (1 - self.distribution[choice, c])
            
            for c in range(self.n):
                pr_have_k[turn + 1, 1, c] = pr_have_k[turn, 0, c] * self.distribution[choice, c] \
                    + pr_have_k[turn, 1, c] * (1 - self.distribution[choice, c])
            
            for c in range(self.n):
                pr_have_k[turn + 1, 2, c] = pr_have_k[turn, 1, c] * self.distribution[choice, c] \
                    + pr_have_k[turn, 2, c]

            # Insert choice and update selected
            self.greedy_alt[turn] = choice
            selected[choice] = True
        
        self.greedy_alt_cost = self.ecost(self.greedy_alt)

if __name__ == "__main__":
    s = SCCP(7)
    # s.print_distribution()
    s.calculate_OPT()
    print("::::::::::::::::::::::::::::::")
    s.find_OPT()
    # s.print_distribution()
    # s.generate_greedy()
    # print("Greedy ordering:")
    # print(s.greedy)
    # print(s.greedy_cost)
    # print("Color choices:")
    # print(s.greedy_color_pick)
    # s.generate_greedy_alt()
    # print("Alt greedy:")
    # print(s.greedy_alt)
    # print(s.greedy_alt_cost)
    # print("Alt greedy color choice:")
    # print(s.greedy_alt_color_pick)