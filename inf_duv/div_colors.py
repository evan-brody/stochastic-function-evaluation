import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 100

def ranking_color(r, g):
    b = 1 - r - g

    if r <= g <= b:
        return 'blue'
    elif r <= b <= g:
        return 'green'
    elif g <= r <= b:
        return 'red'
    elif g <= b <= r:
        return 'black'
    elif b <= r <= g:
        return 'purple'
    elif b <= g <= r:
        return 'yellow'

    raise Exception('?')

xpoints = []
ypoints = []
colors = []
for i, j in itertools.product(range(N), repeat=2):
    if i + j > N: continue
    r = i / N
    g = j / N
    xpoints.append(r)
    ypoints.append(g)
    
    color = ranking_color(r, g)
    colors.append(color)

    plt.plot(r, g, marker='.', color=color)

plt.show()