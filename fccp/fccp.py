import numpy as np
import itertools as it
import copy

class FCCP:
    
    def __init__(self, d, n):
        self.d = d
        self.n = n

        self.distribution = np.empty(shape=(self.n, self.d), dtype=float)
        # self.init_distribution()

        self.NUM_STATES = 2 ** self.d
        self.states_by_size = [ [] for _ in range(self.d + 1) ]
        for j in range(self.NUM_STATES):
            self.states_by_size[j.bit_count()].append(j)
    
    def init_distribution(self):
        for j in range(self.n):
            this_die_distribution = np.random.rand(self.d)
            normalizer = sum(this_die_distribution)
            this_die_distribution[:] /= normalizer

            self.distribution[j] = this_die_distribution
    
    def print_distribution(self):
        for die in self.distribution:
            for color in range(self.d):
                print(round(die[color], 3), end=' ')
            print()
    
    def expected_cost(self, strategy):
        mass_per_state = np.zeros(shape=(self.NUM_STATES,), dtype=float)
        mass_per_state[0] = 1.0
        cost = 0.0

        for j in strategy:
            this_die_distribution = self.distribution[j]

            for size in range(self.d - 1, -1, -1):
                for state in self.states_by_size[size]:
                    cost += mass_per_state[state]
                    mass_lost = 0.0

                    for c in range(self.d):
                        bit_color = 1 << c
                        if state & bit_color: continue

                        state_plus_c = state | bit_color
                        moved_mass = this_die_distribution[c] * mass_per_state[state]
                        mass_per_state[state_plus_c] += moved_mass
                        mass_lost += moved_mass
                    
                    mass_per_state[state] -= mass_lost
            
        return cost

    def generate_OPT(self):
        tests = [ j for j in range(self.n) ]
        self.OPT = np.empty(shape=(self.n,), dtype=int)
        self.EOPT = float('inf')

        self.expected_cost(tests)

        for perm in it.permutations(tests):
            this_perm_cost = self.expected_cost(perm)
            if this_perm_cost < self.EOPT:
                self.EOPT = this_perm_cost
                self.OPT = np.array(perm)

    def print_OPT(self):
        print('OPT:')
        for j in self.OPT:
            print(j, end=' ')
        print()
        print(f'E[OPT]: {self.EOPT}')
    
    def generate_simple_greedy(self):
        max_first_color = 0
        first_color_to_beat = self.distribution[0][0]
        for j in range(1, self.n):
            if self.distribution[j][0] > first_color_to_beat:
                first_color_to_beat = self.distribution[j][0]
                max_first_color = j
        
        self.simple_greedy = np.empty(shape=(self.n,), dtype=int)
        self.simple_greedy[0] = max_first_color

        available = [True] * self.n
        available[max_first_color] = False

        pr_need_c = np.ones(shape=(self.d,), dtype=float) - self.distribution[max_first_color]

        for k in range(1, self.n):
            max_score = float('-inf')
            best_test = None

            for j in range(self.n):
                if not available[j]: continue

                this_test_score = 0.0
                for c in range(self.d):
                    this_test_score += pr_need_c[c] * self.distribution[j][c]
                
                if this_test_score > max_score:
                    max_score = this_test_score
                    best_test = j

            self.simple_greedy[k] = best_test
            available[best_test] = False
            for c in range(self.d):
                pr_need_c[c] *= 1.0 - self.distribution[best_test][c]
        
        self.simple_greedy_cost = self.expected_cost(self.simple_greedy)
    
    def print_simple_greedy(self):
        print('Simple Greedy:')
        for j in self.simple_greedy:
            print(j, end=' ')
        print()
        print(f'E[Simple Greedy]: {self.simple_greedy_cost}')
    
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
    
    def diff(self):
        return self.simple_greedy_cost - self.EOPT
            

GENERATION_SIZE = 100
GENERATION_COUNT = 10_000
PRINT_PER = 100
DN = (3, 6)
if __name__ == '__main__':
    i = 1
    max_diff = float('-inf')
    max_similarity = -1
    max_diff_instance = None
    try:
        for _ in range(10_000):
            fccp = FCCP(*DN)
            fccp.init_distribution()

            fccp.generate_OPT()
            fccp.generate_simple_greedy()

            diff = fccp.diff()
            if diff > max_diff:
                max_diff = diff
                max_diff_instance = copy.deepcopy(fccp)

            if i % PRINT_PER == 0:
                print(f"-------------[{i} -> {round(max_diff,5)}]-------------")
            
            i += 1
        
        for _ in range(GENERATION_COUNT):
            current_parent = copy.deepcopy(max_diff_instance)
            for __ in range(GENERATION_SIZE):
                fccp = FCCP(*DN)
                fccp.init_child_distribution(current_parent.distribution)

                fccp.generate_OPT()
                fccp.generate_simple_greedy()

                diff = fccp.diff()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_instance = copy.deepcopy(fccp)

                if i % PRINT_PER == 0:
                    print(f"-------------[gen {_}, {i} -> {round(max_diff,5)}]-------------")
                
                i += 1

        print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        max_diff_instance.print_simple_greedy()
    except KeyboardInterrupt:
        print("Interrupted."); print()
        max_diff_instance.print_distribution()
        print(f"max diff: {max_diff}"); print()
        max_diff_instance.print_OPT(); print()
        max_diff_instance.print_simple_greedy()