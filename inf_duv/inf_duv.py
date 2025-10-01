# @file     inf_duv.py
# @author   Evan Brody
# @brief    Testbed for infinite coin flipping sequences

import numpy as np
import matplotlib.pyplot as plt

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
        
        return coin_sequence, bias_sequence
            
def plot_sequences(n, p_high=None, p_low=None):
    iuvp = InfUVP(p_high, p_low)
    coin_sequence, bias_sequence = iuvp.generate_sequence(n)

    fig, ax = plt.subplots()
    xticks = list(range(n))

    ax.plot(bias_sequence, marker='o', linestyle='None', color='black')
    ax.set_xticks(xticks)

    # color in the biases
    bar_height = max(bias_sequence) + 5
    for j, bias in zip(xticks, bias_sequence):
        color = 'gray'
        if bias > 1:
            color = 'red'
        elif bias < 1:
            color = 'blue'
        
        ax.bar(j, bar_height, width=1, color=color, alpha=0.3, align='center')

    ax.set_title(f'H = {iuvp.p_high}, T = {iuvp.p_low}')
    ax.set_xlabel('Coin #')
    ax.set_ylabel('Bias')

    plt.show()

if __name__ == '__main__':
    plot_sequences(30, 3/4, 1/8)