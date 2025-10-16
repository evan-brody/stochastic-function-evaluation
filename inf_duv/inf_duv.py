# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite dice rolling sequences

import numpy as np
import itertools as it
import functools as ft
import copy
import sys
import matplotlib.pyplot as plt

N = 100

def point_grad(r1, g1, r2, g2):
    # die 1 is the greedier choice
    b1 = 1 - r1 - g1
    b2 = 1 - r2 - g2

    one_dot_two = b1 * b2 + r1 * r2 + g1 * g2
    if b2 == 1 or r2 == 1 or g2 == 1:
        one_dot_two_inf = float('inf')
    else:
        one_dot_two_inf = b1 / (1 - b2) + r1 / (1 - r2) + g1 / (1 - g2)
    one_dot_self = r1 * r1 + g1 * g1 + b1 * b1
    two_dot_self = r2 * r2 + g2 * g2 + b2 * b2
    
    if r1 == 1 or g1 == 1 or b1 == 1:
        one_inf = float('inf')
    else:
        one_inf = r1 / (1 - r1) + g1 / (1 - g1) + b1 / (1 - b1)
    
    if r1 == 1 or g1 == 1 or b1 == 1:
        two_dot_one_inf = float('inf')
    else:
        two_dot_one_inf = r2 / (1 - r1) + g2 / (1 - g1) + b2 / (1 - b1)

    if one_dot_self <= one_dot_two and one_inf <= two_dot_one_inf:
        return 'red'
    elif one_dot_self <= one_dot_two:
        return 'yellow'
    elif one_inf <= two_dot_one_inf:
        return 'orange'
    else:
        return 'green'

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
        strategies = list(it.product(range(self.k), repeat=self.n))
        strategies = strategies[::-1]
        for strategy in strategies:
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
        for j in range(self.k):
            new_die = copy.deepcopy(parent_distribution[j])
            scale_vector = self.get_scale_vector()
            for c in range(self.d):
                new_die[c] *= scale_vector[c]
            
            new_die = self.clamp(new_die)
            new_die = self.normalize(new_die)

            self.distribution[j] = copy.deepcopy(new_die)
        
        return self.distribution
    
    def dz_helper(self, j):
        if not (j == 0 or j == 1): return True

        fixed_die = self.distribution[j]
        other_die = self.distribution[abs(1 - j)]

        other_dot_fixed = 0
        fixed_dot_self = 0
        for c in range(self.d):
            other_dot_fixed += other_die[c] * fixed_die[c]
            fixed_dot_self += fixed_die[c] * fixed_die[c]
        
        other_dot_fixed_inf = 0
        fixed_inf = 0
        for c in range(self.d):
            if fixed_die[c] == 1:
                other_dot_fixed_inf = float('inf')
                fixed_inf = float('inf')
            else:
                other_dot_fixed_inf += other_die[c] / (1.0 - fixed_die[c])
                fixed_inf += fixed_die[c] / (1.0 - fixed_die[c])

        return (other_dot_fixed <= fixed_dot_self) != (other_dot_fixed_inf <= fixed_inf)

    def in_danger_zone(self):
        if self.k != 2: return True

        return self.dz_helper(0) or self.dz_helper(1)

    def plot_two_dice(self):
        if self.k != 2: return

        r1 = self.distribution[0][0]
        g1 = self.distribution[0][1]

        xpoints = []
        ypoints = []

        for i, j in it.product(range(N), repeat=2):
            if i + j > N: continue
            r2 = i / N
            g2 = j / N

            xpoints.append(r2)
            ypoints.append(g2)

            color = point_grad(r1, g1, r2, g2)

            plt.plot(r2, g2, marker='.', color=color)
    
        plt.plot(r1, g1, marker='o', color='black')
        plt.plot(self.distribution[1][0], self.distribution[1][1], marker='o', color='black')
        plt.show()
    
    # Conjectured condition for using b greedily
    def weak_condition_asymmetric(self, flip=False):
        if self.k != 2:
            raise Exception("Weak condition only applies when k = 2.")

        if not flip:
            a = self.distribution[0]
            b = self.distribution[1]
        else:
            a = self.distribution[1]
            b = self.distribution[0]

        a_dot_a = sum([ a[c] * a[c] for c in range(self.d) ])
        a_dot_b = sum([ a[c] * b[c] for c in range(self.d) ])

        if a_dot_b >= a_dot_a:
            return False
        
        for c in range(self.d):
            if a[c] == 1: return True
        
        a_dot_a_inf = sum([ a[c] * (a[c] / (1.0 - a[c])) for c in range(self.d) ])
        b_dot_a_inf = sum([ b[c] * (a[c] / (1.0 - a[c])) for c in range(self.d) ])

        if b_dot_a_inf >= a_dot_a_inf:
            return False
        
        return True
    
    def weak_condition_symmetric(self):
        return self.weak_condition_asymmetric() and self.weak_condition_asymmetric(True)

# Plots a map of whether or not greedy is optimal
# TODO: check strong condition instead
def plot_greedy_map(r, g):
    d = 3; n = 6; k = 2
    duv = DUV(d, n, k)
    duv.distribution[0][0] = r
    duv.distribution[0][1] = g
    duv.distribution[0][2] = 1.0 - r - g

    for i, j in it.product(range(N), repeat=2):
        if i + j > N: continue
        r2 = i / N
        g2 = j / N

        duv.distribution[1][0] = r2
        duv.distribution[1][1] = g2
        duv.distribution[1][2] = 1.0 - r2 - g2

        duv.generate_greedy()
        duv.brute_force_OPT()

        # condition = duv.weak_condition_symmetric()
        # greedy_optimal = not (duv.greedy_cost > duv.EOPT)

        # if condition and not greedy_optimal:
        #     plt.plot(r2, g2, marker='.', color='red')
        # else:
        #     plt.plot(r2, g2, marker='.', color='green')

        # ab = duv.weak_condition_asymmetric()
        # ba = duv.weak_condition_asymmetric(True)

        # if ab and ba:
        #     plt.plot(r2, g2, marker='.', color='purple')
        # elif ab:
        #     plt.plot(r2, g2, marker='.', color='red')
        # elif ba:
        #     plt.plot(r2, g2, marker='.', color='blue')
        # else:
        #     plt.plot(r2, g2, marker='.', color='yellow')

        # Condition map
        # if duv.weak_condition_symmetric():
        #     plt.plot(r2, g2, marker='.', color='green')
        # else:
        #     plt.plot(r2, g2, marker='.', color='red')

        # Greedy optimal map
        if duv.greedy_cost > duv.EOPT:
            plt.plot(r2, g2, marker='.', color='red')
        else:
            plt.plot(r2, g2, marker='.', color='green')

    plt.plot(r, g, marker='o', color='black')
    plt.title(f"d={d}; n={n}; k={k}")

GENERATION_SIZE = 1000
GENERATION_COUNT = 100
DNK = (3, 8, 2)
if __name__ == '__main__':

    r = 1/4
    g = 1/5
    plot_greedy_map(r, g)
    plt.show()
    sys.exit(0)

    # duv = DUV(3, 12, 2)

    # duv.distribution[0][0] = r
    # duv.distribution[0][1] = g
    # duv.distribution[0][2] = 1.0 - r - g

    # duv.distribution[1][0] = 1/3
    # duv.distribution[1][1] = 1/3
    # duv.distribution[1][2] = 1/3

    # duv.brute_force_OPT()
    # print(duv.OPT)
    # sys.exit(0)

    max_diff = float('-inf')
    i = 1
    for _ in range(100_000):
        duv = DUV(*DNK)
        duv.init_distribution()

        duv.brute_force_OPT()
        duv.generate_greedy()

        diff = duv.greedy_cost - duv.EOPT
        # if diff > 0 and duv.weak_condition_symmetric():
        #     duv.plot_two_dice()
        #     print(diff); print()
        #     print(duv.distribution); print()
        #     duv.print_OPT(); print()
        #     duv.print_greedy(); print()
        #     sys.exit(0)

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
    print(max_diff_instance.distribution); print()
    max_diff_instance.print_OPT(); print()
    max_diff_instance.print_greedy(); print()