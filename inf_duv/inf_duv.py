# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite coin flipping sequences

import numpy as np
import matplotlib.pyplot as plt
import math

class InfUVP:
    def __init__(self, p_high=None, p_low=None):
        self.generate_coins()

        if p_high is not None:
            self.p_high = p_high
        
        if p_low is not None:
            self.p_low = p_low

    # Guarantees that coins are on both sides of 1/2
    # Other case isn't interesting
    def generate_coins(self):
        eps_heads = np.random.rand() * 0.5
        eps_tails = np.random.rand() * 0.5

        self.p_high = 0.5 + eps_heads
        self.p_low = 0.5 - eps_tails
    
    def generate_sequence(self, n):
        head_bias = 0.5
        tail_bias = 0.5

        bias_sequence = np.empty(shape=(n,), dtype=float)
        coin_sequence = np.empty(shape=(n,), dtype=bool) # True = heads, False = tails

        # NOTE: this is only optimal because both coins are guaranteed
        #       to be on either side of 1/2
        for k in range(n):
            bias_sequence[k] = head_bias
            if tail_bias >= head_bias:
                coin_sequence[k] = True

                head_bias *= self.p_high
                tail_bias *= 1 - self.p_high
            else:
                coin_sequence[k] = False

                head_bias *= self.p_low
                tail_bias *= 1 - self.p_low
            
            normalizer = head_bias + tail_bias
            head_bias /= normalizer
            tail_bias /= normalizer
        
        return coin_sequence, bias_sequence
            
def plot_sequences(n, p_high=None, p_low=None):
    iuvp = InfUVP(p_high, p_low)
    coin_sequence, bias_sequence = iuvp.generate_sequence(n)

    fig, ax = plt.subplots()
    xticks = list(range(n))

    ax.plot(bias_sequence, marker='o', linestyle='None', color='black')
    ax.set_xticks(xticks)

    # color in the biases
    bar_height = 1
    for j, bias in zip(xticks, bias_sequence):
        color = 'gray'
        if bias < 0.5: # chose heads
            color = 'red'
        elif bias > 0.5: # chose tails
            color = 'blue'
        
        ax.bar(j, bar_height, width=1, color=color, alpha=0.3, align='center')

    ax.set_title(f'H = {iuvp.p_high}, T = {iuvp.p_low}')
    ax.set_xlabel('Coin #')
    ax.set_ylabel('Bias')

    plt.show()

if __name__ == '__main__':
    plot_sequences(100, 0.75, 0.125)