# Counterexample to optimality for the greedy adaptive algorithm for the conservative case of SCCP
from mpmath import mp
import numpy as np

# Configure mp
mp.dps = 5     # Decimal places used by mp.mpf
mp.pretty = True # Turn pretty-printing on

die1b = mp.mpf(.7)
die1g = mp.mpf(.1)

die2b = mp.mpf(.6)
die2g = mp.mpf(.2)

die3b = mp.mpf(.5)
die3g = mp.mpf(.3)

die4b = mp.mpf(.4)
die4g = mp.mpf(.4)

dice = [ (die1b, die1g), (die2b, die2g), (die3b, die3g), (die4b, die4g) ]

BLUE = 0
GREEN = 1

def ec_find_one(dice: list, color):
    die_count = len(dice)
    match die_count:
        case 0:
            return mp.mpf(0), None
        case 1:
            return mp.mpf(1), dice[0]

    # Sort in decreasing order of Pr[=color]
    dice.sort(key=lambda x: x[color], reverse=True)

    ecost = mp.mpf(1) # E[cost] >= 1

    # Chance we move on to the next test
    prod = mp.mpf(1) - dice[0][color]

    for i in range(1, len(dice)):
        ecost += prod
        prod *= mp.mpf(1) - dice[i][color]
    
    return ecost, dice[0]

def ec_find_two(dice):
    die_count = len(dice)
    match die_count:
        case 0:
            return mp.mpf(0), None
        case 1:
            return mp.mpf(1), dice[0]
        case 2:
            return mp.mpf(2), dice[1]
    
    best_die = None
    best_cost = float('inf')
    best_second_die = None
    dice_costs = [0] * 4
    for i in range(die_count):
        this_die = dice[i]

        not_this_die = [ d for d in dice if d != this_die ]
        phase_two_result = ec_find_two(not_this_die)
        blue_result = ec_find_one(not_this_die, BLUE)
        green_result = ec_find_one(not_this_die, GREEN)

        pr_phase_two = mp.mpf(1) - this_die[BLUE] - this_die[GREEN]

        this_die_cost = mp.mpf(1)
        this_die_cost += phase_two_result[0] * pr_phase_two
        this_die_cost += blue_result[0] * this_die[GREEN]
        this_die_cost += green_result[0] * this_die[BLUE]
        
        if die_count == 4:
            dice_costs[i] = (this_die_cost, phase_two_result[0], blue_result[0], green_result[0])

        if this_die_cost < best_cost:
            best_cost = this_die_cost
            best_die = this_die
            if phase_two_result[1] is not None:
                best_second_die = phase_two_result[1]

    if die_count == 4:
        print("bsd", best_second_die)
        print(dice_costs)
    return best_cost, best_die

res = ec_find_two(dice)

print(res)

ph2 = mp.mpf(.48) * mp.mpf(2) + mp.mpf(.5) * mp.mpf(1.96) + mp.mpf(.02) * mp.mpf(1.4) + mp.mpf(1)
G = mp.mpf(1) + mp.mpf(.96) + mp.mpf(.96) * mp.mpf(.98)
B = mp.mpf(1) + mp.mpf(.4) + mp.mpf(.4) * mp.mpf(.5)

total = mp.mpf(.62) * ph2 + mp.mpf(.3) * G + mp.mpf(.08) * B + mp.mpf(1)

print(total)