# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite coin flipping sequences

import numpy as np
import matplotlib.pyplot as plt
import math
from mpmath import mp

# Configure mp
mp.dps = 100     # Decimal places used by mp.mpf
mp.pretty = True # Turn pretty-printing on

class InfUVP:
    def __init__(self, p_high=None, p_low=None):
        self.generate_coins()

        if p_high is not None:
            self.p_high = mp.mpf(p_high)
        
        if p_low is not None:
            self.p_low = mp.mpf(p_low)

    # Guarantees that coins are on both sides of 1/2
    # Other case isn't interesting
    def generate_coins(self):
        eps_heads = np.random.rand() * 0.5
        eps_tails = np.random.rand() * 0.5

        self.p_high = mp.mpf(0.5 + eps_heads)
        self.p_low = mp.mpf(0.5 - eps_tails)
    
    def generate_sequence(self, n):
        head_bias = mp.mpf(0.5)
        tail_bias = mp.mpf(0.5)

        self.bias_sequence = []

        # NOTE: this is only optimal because both coins are guaranteed
        #       to be on either side of 1/2
        for _ in range(n):
            self.bias_sequence.append(head_bias)
            if tail_bias >= head_bias:
                head_bias *= self.p_high
                tail_bias *= 1 - self.p_high
            else:
                head_bias *= self.p_low
                tail_bias *= 1 - self.p_low
            
            normalizer = head_bias + tail_bias
            head_bias /= normalizer
            tail_bias /= normalizer
        
        return self.bias_sequence

    # TODO: sliding window period check

    def get_block_length_sequence(self):
        oeis = []
        current_length = 0
        is_heads = True
        for i in range(len(self.bias_sequence)):
            if self.bias_sequence[i] < 0.5 and is_heads:
                is_heads = False
                oeis.append(current_length)
                current_length = 1
            elif self.bias_sequence[i] > 0.5 and not is_heads:
                is_heads = True
                oeis.append(current_length)
                current_length = 1
            else:
                current_length += 1
        
        self.oeis = oeis

        with open('output.txt', 'w') as f:
            print(self.oeis, file=f)

            
def plot_sequences(n, p_high=None, p_low=None):
    iuvp = InfUVP(p_high, p_low)
    bias_sequence = iuvp.generate_sequence(n)

    # slow check for repeated biases
    # TODO: optimize this

    min_ij = (None, None)
    min_diff = mp.mpf('inf')
    for i, bias in enumerate(bias_sequence):
        for j in range(i + 1, len(bias_sequence)):
            diff = abs(bias - bias_sequence[j])
            if diff < min_diff:
                min_ij = (i, j)
                min_diff = diff

            if diff <= np.finfo(float).eps:
                print(f"Biases repeat at {i}, {j}")

    # printing

    print(f"Minimum difference: {min_diff} between {min_ij}")

    iuvp.get_block_length_sequence()
    print(iuvp.oeis)

    # for i, bias in enumerate(bias_sequence):
    #     print(f"{i + 1}: {bias}")

    # plotting

    fig, ax = plt.subplots()
    xticks = list(range(n))

    ax.plot(bias_sequence, marker='o', linestyle='None', color='black')
    ax.set_xticks(xticks)

    # color in the biases
    bar_height = 1
    for j, bias in zip(xticks, bias_sequence):
        color = 'gray'
        if bias > 0.5: # chose tails
            color = 'red'
        elif bias < 0.5: # chose heads
            color = 'blue'
        
        ax.bar(j, bar_height, width=1, color=color, alpha=0.3, align='center')

    ax.set_title(f'H = {round(iuvp.p_high, 5)}, T = {round(iuvp.p_low, 5)}')
    ax.set_xlabel('Coin #')
    ax.set_ylabel('Bias')

    plt.show()

if __name__ == '__main__':
    plot_sequences(200, 0.7, 0.35)