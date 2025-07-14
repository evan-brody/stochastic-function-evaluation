# @file     sccp_dup.py
# @author   Evan Brody
# @brief    Simulates the d = n case of the Stochastic Coupon Collection Problem

import numpy as np
import itertools as it
import copy

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
    
    # Brute-force search for the optimal permutation
    def find_OPT(self):
        self.OPT = None
        self.EOPT = SCCP_float(self.n)
        for permutation in it.permutations(np.array([ i for i in range(self.n) ])):
            this_cost = self.ecost(permutation)

            if this_cost <= self.EOPT:
                self.EOPT = this_cost
                self.OPT = permutation

        print([ int(die) for die in self.OPT ])
        print(self.EOPT)
        
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
        pr_have_k[0, 0] = 1

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
                this_color_score += self.ecost_color_get_one(c, queues[c], selected) * pr_have_k[turn, 1, c]
                this_color_score += self.ecost_color_get_two(c, queues[c], selected) * pr_have_k[turn, 0, c]

                if this_color_score < color_choice_score:
                    color_choice_score = this_color_score
                    color_choice = c
            
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

if __name__ == "__main__":
    s = SCCP(7)
    s.print_distribution()
    s.generate_greedy()
    print(s.greedy)
    print(s.greedy_cost)
    s.find_OPT()