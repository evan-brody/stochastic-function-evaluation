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
    
    def ecost_color(self, color):
        sc = copy.deepcopy(self.distribution)
        
    
    def generate_greedy(self):
        self.greedy = np.empty(self.n, int)


if __name__ == "__main__":
    s = SCCP(7)
    s.find_OPT()
    s.print_distribution()