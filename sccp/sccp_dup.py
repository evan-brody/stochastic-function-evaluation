# @file     sccp_dup.py
# @author   Evan Brody
# @brief    Simulates the d = n case of the Stochastic Coupon Collection Problem

import numpy as np
import itertools as it
import copy
from mpmath import mp
import functools as ft

# Configure mp
mp.dps = 5          # Decimal places used by mp.mpf
mp.pretty = True    # Turn pretty-printing on

SCCP_float = np.float64

class SCCP:
    def __init__(self, n):
        self.n = n
        self.init_distribution()
    
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
                cost_with_best_end_test = SCCP_float(self.n + 1)

                # Pulling test j out of subset S creates a prefix
                for j, singleton in enumerate(singletons):
                    # j must be in S
                    if not (singleton & subset): continue

                    # Bitwise XOR removes j from S
                    no_j = subset ^ singleton

                    # E[cost(S)] = E[cost(S\{j})] + Pr[test j]
                    # Pr[test j] = Pr[all colors in prefix are unique]
                    # We sum over disjoint outcomes to calculate this value                    
                    cost_with_end_j = optimal_costs[no_j] + sum(pr_unique_outcomes[no_j])
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
        
        # Bit representation of [n]
        ALL_TESTS = NUM_TEST_SETS - 1

        self.OPT = optimal_permutations[ALL_TESTS]
        self.EOPT = optimal_costs[ALL_TESTS]
    
    def print_OPT(self):
        print("OPT:", [ int(j) for j in self.OPT ])
        print("E[OPT]:", self.EOPT)
        for c in range(self.n):
            for j in self.OPT:
                print(round(self.distribution[j][c], 3), end='\t')
            print()
        
    def ecost_color_get_one(self, color, queue, selected):
        E = SCCP_float(1)

        product = SCCP_float(1)
        for test in queue[:self.n - 1]:
            if selected[test]: continue
            product *= self.distribution[test][color]
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

            E += pr_have_k[num_tested][0] + pr_have_k[num_tested][1]

            pr_have_k[num_tested + 1][0] = pr_have_k[num_tested][0] * (1 - self.distribution[test][color])
            pr_have_k[num_tested + 1][1] = pr_have_k[num_tested][0] * self.distribution[test][color] \
                + pr_have_k[num_tested][1] * (1 - self.distribution[test][color])

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
                
                pr_sum = pr_have_k[turn][1][c] + pr_have_k[turn][0][c]
                one_portion = pr_have_k[turn][1][c] / pr_sum
                zero_portion = pr_have_k[turn][0][c] / pr_sum
                this_color_score += self.ecost_color_get_one(c, queues[c], selected) * one_portion
                this_color_score += self.ecost_color_get_two(c, queues[c], selected) * zero_portion

                if this_color_score < color_choice_score:
                    color_choice_score = this_color_score
                    color_choice = c
            
            self.greedy_color_pick[turn] = color_choice

            choice = None
            i = 0
            while i < self.n:
                if not selected[queues[color_choice][i]]:
                    choice = queues[color_choice, i]
                    break
                i += 1

            if i == self.n:
                raise Exception("Error in greedy generation.")

            # Update probabilities
            for c in range(self.n):
                pr_have_k[turn + 1][0][c] = pr_have_k[turn][0][c] * (1 - self.distribution[choice][c])
            
            for c in range(self.n):
                pr_have_k[turn + 1][1][c] = pr_have_k[turn][0][c] * self.distribution[choice][c] \
                    + pr_have_k[turn][1][c] * (1 - self.distribution[choice][c])
            
            for c in range(self.n):
                pr_have_k[turn + 1][2][c] = pr_have_k[turn][1][c] * self.distribution[choice][c] \
                    + pr_have_k[turn][2][c]

            # Insert choice and update selected
            self.greedy[turn] = choice
            selected[choice] = True
        
        self.greedy_cost = self.ecost(self.greedy)

    def dprod(self, i, j):
        irow = self.distribution[i]
        jrow = self.distribution[j]

        res = SCCP_float(0)
        for k in range(self.n):
            res += irow[k] * jrow[k]
        
        return res

    def generate_backwards_greedy(self):
        self.bgreedy = np.empty(self.n, dtype=int)
        selected = np.array([False] * self.n)

        for turn in range(self.n - 1, -1, -1):
            unused_tests = list(filter(lambda j: not selected[j], range(self.n)))

            best_candidate = None
            best_candidate_score = SCCP_float(self.n)
            for candidate in unused_tests:
                this_candidate_score = SCCP_float(0)
                for other_test in unused_tests:
                    if other_test == candidate: continue
                    this_candidate_score += self.dprod(candidate, other_test)
                
                if this_candidate_score < best_candidate_score:
                    best_candidate_score = this_candidate_score
                    best_candidate = candidate
                    # print(best_candidate)
            
            self.bgreedy[turn] = best_candidate
            selected[best_candidate] = True
        
        self.bgreedy_cost = self.ecost(self.bgreedy)
    
    def test_score(self, j, used_tests):
        score = SCCP_float(0)
        
        for color_set in it.combinations(range(self.n), len(used_tests)):
            pr_color_set = SCCP_float(0)

            for color_assignment in it.permutations(color_set):
                pr_assignment = SCCP_float(1)

                for k, color in enumerate(color_assignment):
                    pr_assignment *= self.distribution[used_tests[k]][color]
                
                pr_color_set += pr_assignment
            
            pr_j_helpful = SCCP_float(0)
            for helpful_color in color_set:
                pr_j_helpful += self.distribution[j][helpful_color]
            
            score += pr_color_set * pr_j_helpful
        
        return score

    def generate_exact_greedy(self):
        self.exact_greedy = np.empty(self.n, dtype=int)
        
        best_first_test = None
        best_first_test_ecost = SCCP_float(self.n + 1)
        for first_test in range(self.n):
            self.exact_greedy_first_test(first_test)

            this_ecost = self.ecost(self.exact_greedy)
            if this_ecost < best_first_test_ecost:
                best_first_test_ecost = this_ecost
                best_first_test = first_test
        
        self.exact_greedy_first_test(best_first_test)
        self.exact_greedy_cost = self.ecost(self.exact_greedy)

    def print_exact_greedy(self):
        print("Exact Greedy:")
        print([ int(j) for j in self.exact_greedy ])
        print("E[Exact Greedy]:", self.exact_greedy_cost)
        print("Approx. factor:", round(self.exact_greedy_cost / self.EOPT, 4))
        for c in range(self.n):
            for j in self.exact_greedy:
                print(round(self.distribution[j][c], 3), end='\t')
            print()

    def exact_greedy_first_test(self, first_test):
        self.exact_greedy[0] = first_test

        selected = np.array([ j == first_test for j in range(self.n) ])

        used_tests = [ first_test ]

        for k in range(1, self.n):
            unused_tests = [ j for j in range(self.n) if not selected[j] ]

            best_test = None
            best_test_score = SCCP_float(0)
            for candidate in unused_tests:
                this_test_score = self.test_score(candidate, used_tests)
                if this_test_score > best_test_score:
                    best_test_score = this_test_score
                    best_test = candidate
                
            self.exact_greedy[k] = best_test
            selected[best_test] = True
            used_tests.append(best_test)


    def generate_forward_greedy(self):
        self.fgreedy = np.empty(self.n, dtype=int)
        selected = np.array([False] * self.n)

        # Find max pair
        max_pair = None
        max_pair_score = SCCP_float(0)

        for i, j in it.combinations(range(self.n), 2):
            this_pair_score = self.dprod(i, j)
            if this_pair_score > max_pair_score:
                max_pair_score = this_pair_score
                max_pair = (i, j)
        
        self.fgreedy[0] = max_pair[0]
        self.fgreedy[1] = max_pair[1]

        selected[max_pair[0]] = selected[max_pair[1]] = True

        used_tests = list(max_pair)

        for k in range(2, self.n):
            unused_tests = [ j for j in range(self.n) if not selected[j] ]

            best_test = None
            best_test_score = SCCP_float(0)

            for candidate in unused_tests:
                this_candidate_score = SCCP_float(0)
                for previous_test in used_tests:
                    this_candidate_score += self.dprod(candidate, previous_test)
                
                if this_candidate_score > best_test_score:
                    best_test_score = this_candidate_score
                    best_test = candidate
            
            self.fgreedy[k] = best_test
            used_tests.append(best_test)
            selected[best_test] = True
        
        self.fgreedy_cost = self.ecost(self.fgreedy)
    
    def print_forward_greedy(self):
        print("Forward Greedy:")
        print([ int(j) for j in self.fgreedy ])
        print("E[Forward Greedy]:", s.fgreedy_cost)
        print("Approx. factor:", round(s.fgreedy_cost / s.EOPT, 4))

if __name__ == "__main__":
    for _ in range(100):
        s = SCCP(8)
        s.calculate_OPT()
        # s.print_OPT()

        print()
        s.generate_exact_greedy()
        # s.print_exact_greedy()

        if abs(s.exact_greedy_cost - s.EOPT) > 0.001:
            s.print_OPT()
            s.print_exact_greedy()

        print("::::::::::::::::::::::::::::::::::::::::::::")
