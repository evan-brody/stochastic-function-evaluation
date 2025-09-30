# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite coin flipping

import numpy as np

class InfUVP:
    def __init__(self):
        self.generate_coins()

    # Guarantees that coins are on both sides of 1/2
    # Other case isn't interesting
    def generate_coins(self):
        eps_heads = np.random.rand() * 0.5
        eps_tails = np.random.rand() * 0.5

        self.p_high = 0.5 + eps_heads
        self.p_low = 0.5 - eps_tails
    
    def generate_sequence(self, n):
        # Need to track ratio instead of individual biases for numerical stability
        # H / T
        bias_ratio = 1

        bias_sequence = np.empty(shape=(n,), dtype=float)
        coin_sequence = np.empty(shape=(n,), dtype=bool) # True = heads, False = tails

        # NOTE: this is only optimal because both coins are guaranteed
        #       to be on either side of 1/2
        for k in range(n):
            bias_sequence[k] = bias_ratio
            if bias_ratio <= 1:
                coin_sequence[k] = True

                bias_ratio *= self.p_high
                bias_ratio /= 1 - self.p_high
            elif bias_ratio > 1:
                coin_sequence[k] = False

                bias_ratio *= self.p_low
                bias_ratio /= 1 - self.p_low
            

            


