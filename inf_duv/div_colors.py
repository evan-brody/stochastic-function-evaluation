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

def point_grad(r1, g1, r2, g2):
    # die 1 is the greedier choice
    b1 = 1 - r1 - g1
    b2 = 1 - r2 - g2

    one_dot_two = b1 * b2 + r1 * r2 + g1 * g2
    if b2 == 1 or r2 == 1 or g2 == 1:
        one_dot_inf_two = float('inf')
    else:
        one_dot_inf_two = (b1 * b2) / (1 - b2) + (r1 * r2) * (1 - r2) + (g1 * g2) / (1 - g2)
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
        return 'green'
    elif one_dot_self <= one_dot_two:
        return 'yellow'
    elif one_inf <= two_dot_one_inf:
        return 'orange'
    else:
        return 'red'

def plot_ranking_color():
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

def plot_point_grad():
    r1 = np.random.rand() / 2
    g1 = np.random.rand() / 2

    xpoints = []
    ypoints = []
    colors = []

    for i, j in itertools.product(range(N), repeat=2):
        if i + j > N: continue
        r2 = i / N
        g2 = j / N

        xpoints.append(r2)
        ypoints.append(g2)

        color = point_grad(r1, g1, r2, g2)
        colors.append(color)

        plt.plot(r2, g2, marker='.', color=color)
    
    plt.plot(r1, g1, marker='o', color='black')
    plt.show()

if __name__ == '__main__':
    plot_point_grad()