// @file    sccpSimulation.cpp
// @author  Evan Brody
// @brief   Environment for simulating (nonadaptive) strategies for the stochastic coupon collection problem
//          Assumed unit-cost everywhere

#include <random>
#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <exception>

// #define DEBUG
// #define PRINT_MARKOV
// #define PRINT_DICE_DISTS

constexpr std::uint64_t D = 3;      // Number of coupons
constexpr std::uint64_t N = 20;     // Number of tests

// When we're searching for the optimal strategy, we print our progress
//      every OPT_SEARCH_PRINT permutations checked
// Making it a power of 2 minus one allows for fast modulo via bitwise AND
constexpr std::uint64_t OPT_SEARCH_PRINT = (1 << 20) - 1;

#ifdef PRINT_MARKOV
constexpr std::string_view WHITESPACE = "\t";
#else
constexpr std::string_view WHITESPACE = " ";
#endif

typedef long double SCCPFloat;

// Converts a state represented as a bitstring to a state represented as a string
// Only configured for D <= 4
// @param   state   The state represented as a bitstring
// @return          The state represented as a string
std::string bitToStateString(std::uint64_t state) {
    if (!(D <= 4)) { return ""; }

    std::string stateString;
    for (std::size_t c = 0; c < D; ++c) {
        if ((1 << c) & state) {
            switch (c) {
                case 0:
                    stateString += "R";
                    break;
                case 1:
                    stateString += "G";
                    break;
                case 2:
                    stateString += "B";
                    break;
                case 3:
                    stateString += "Y";
                    break;
                default:
                    stateString += "?";
            }
        }
    }

    return stateString;
}

// Prints a number rounded to 2 decimal places
// @param   os  The stream to print to
// @param   num The number to print
// @return      The stream that was passed in, now modified
std::ostream& roundPrint(std::ostream& os, SCCPFloat num) {
    os << std::round(num * 10000.0f) * 0.0001f;

    return os;
}

// Log base 2 with inline x86 assembly
// https://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
static inline std::uint64_t log2(const std::uint64_t x) {
  std::uint64_t y;
  asm ("\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y;
}

// https://stackoverflow.com/questions/8871204/count-number-of-1s-in-binary-representation
static inline std::uint64_t bitCount(std::uint64_t u) {
    std::uint64_t uCount;

    uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
    return ((uCount + (uCount >> 3)) & 030707070707) & 63;
}

// Naive factorial function
static inline constexpr std::uint64_t factorial(std::uint64_t n) {
    std::uint64_t res = 1;
    for (std::size_t i = 2; i <= n; ++i) {
        res *= i;
    }

    return res;
}

class SCCP {
    // Represents a nonadaptive strategy
    struct NAStrategy {
        std::uint64_t order[N];
        SCCPFloat cost = N + 1;
    };
public:
    SCCP(bool highVariance = false) : isHighVariance(highVariance) {
        if (highVariance) {
            initDistributionHighVariance();
        } else {
            initDistribution();
        }

        initStates();
    }

    // Resets the state of the instance, including generating a fresh distribution
    void reset() {
        for (std::size_t i = 0; i < N; ++i) {
            OPT.order[i] = N;
            localOPT.order[i] = N;
            greedy.order[i] = N;
        }

        OPT.cost = N + 1;
        localOPT.cost = N + 1;
        greedy.cost = N + 1;
    
        if (isHighVariance) {
            initDistributionHighVariance();
        } else {
            initDistribution();
        }
    }

    // We maintain an array of possible states, where a bit being 1 means we have a color in that state
    // We need to initialize and sort these states by number of colors obtained so that our printed info looks nicer
    void initStates() {
        constexpr std::uint64_t numStates = 1 << D;
        for (std::size_t state = 0; state < numStates; ++state) {
            states[state] = state;
        }

        // Sort states by number of colors obtained
        std::sort(
            states,
            states + numStates,
            [this](std::uint64_t a, std::uint64_t b) {
                return bitCount(a) < bitCount(b);
            }
        );
    }

    // Generates a distribution over the tests and colors
    void initDistribution() {
        std::random_device rd; // Seed for Mersenne Twister
        std::mt19937 mersenne(rd()); // Mersenne Twister
        std::uniform_real_distribution pdf(0.0f, 1.0f); // Uniform on [0, 1]

        for (std::size_t i = 0; i < N; ++i) {
            // To generate the distribution for a particular die, we will generate (d - 1) "markers" in [0, 1]
            // The space between markers will be the probability assigned to colors
            for (std::size_t c = 0; c < D - 1; ++c) {
                distribution[i][c] = pdf(mersenne);
            }
            distribution[i][D - 1] = 1.0f;

            // Entry D - 1 is already sorted, so don't include
            std::sort(distribution[i], distribution[i] + D - 1);

            // Calculate the space between markers
            // No need to change entry 0
            for (std::size_t c = D; c-- > 1;) {
                distribution[i][c] -= distribution[i][c - 1];
            }
        }
    }
    
    void initDistributionHighVariance() {
        std::random_device rd; // Seed for Mersenne Twister
        std::mt19937 mersenne(rd()); // Mersenne Twister

        std::uniform_int_distribution<std::uint64_t> uniformDice(0, D);
        std::uniform_real_distribution uniform01(0.0f, 1.0f);

        for (std::size_t die = 0; die < N; ++die) {
            std::size_t biasedDie = uniformDice(mersenne);
            if (biasedDie == D) {
                for (std::size_t c = 0; c < D - 1; ++c) {
                    distribution[die][c] = uniform01(mersenne);
                }
                distribution[die][D - 1] = 1.0f;

                // Entry D - 1 is already sorted, so don't include
                std::sort(
                    distribution[die],
                    distribution[die] + D - 1
                );

                // Calculate the space between markers
                // No need to change entry 0
                for (std::size_t c = D; c-- > 1;) {
                    distribution[die][c] -= distribution[die][c - 1];
                }
            } else {
                distribution[die][biasedDie] = pow(uniform01(mersenne), 0.75f);

                SCCPFloat otherProbs[D - 1];
                for (std::size_t c = 0; c < D - 2; ++c) {
                    otherProbs[c] = uniform01(mersenne);
                }
                otherProbs[D - 2] = 1.0f;

                std::sort(
                    otherProbs,
                    otherProbs + D - 1
                );

                for (std::size_t c = D; c-- > 1;) {
                    otherProbs[c] -= otherProbs[c - 1];
                }

                SCCPFloat scale = 1.0f - distribution[die][biasedDie];
                std::size_t otherProbsIndex = 0;
                for (std::size_t c = 0; c < D; ++c) {
                    if (c == biasedDie) { continue; }

                    distribution[die][c] = otherProbs[otherProbsIndex] * scale;
                    ++otherProbsIndex;
                }
            }
        }
    }

    // Prints information about the distribution over the tests and colors
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printDistribution(std::ostream& os) const {
        // Rows are dice, columns are colors
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t c = 0; c < D; ++c) {
                roundPrint(os, distribution[i][c]) << '\t';
            }
            os << '\n';
        }

        return os;
    }

    // Calculates the expected cost of a nonadaptive strategy
    // @param   order   The strategy to evaluate
    // @return          The strategy's expected cost
    SCCPFloat expectedCost(const std::uint64_t* order) const {
        // There will be 2^d states in our Markov chain
        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;

        // Represents the state of the system just after test i
        //      where no tests done is considered just after the 0th test
        // Only goes up to N - 1 since we don't care about the state of the system
        //      after the Nth test
        SCCPFloat stateVector[N][numStates];

        stateVector[0][0] = 1.0f; // We begin in state 0 (no colors collected)
        for (std::size_t i = 1; i < numStates; ++i) {
            stateVector[0][i] = 0.0f;
        }

        SCCPFloat E = 1.0f; // Must do first test
        for (std::size_t i = 1; i < N; ++i) {
            std::uint64_t test = order[i - 1];

            stateVector[i][0] = 0.0f; // No chance we have no colors after step 1
            for (std::size_t state = 1; state < numStates; ++state) {
                stateVector[i][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = (1 << c);
                    std::uint64_t gainedColorBit = colorBit & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = colorBit ^ state;

                    // Pr[got this color on this roll
                    //    OR had this color already, got it on this roll]
                    stateVector[i][state] += 
                    distribution[test][c] * (stateVector[i - 1][stateMinusColor] + stateVector[i - 1][state]);
                }

                if (state != stateFinished) {
                    E += stateVector[i][state];
                }
            }
        }

        #ifdef DEBUG
        for (std::size_t state = 0; state < numStates; ++state) {
            for (std::size_t i = 0; i < N + 1; ++i) {
                roundPrint(std::cout, stateVector[i][state]) << '\t';
            }
            std::cout << '\n';
        }
        #endif

        return E;
    }

    // Prints the Markov chain progression of the given order
    // @param   os      The stream to print to
    // @param   order   The permutation to calculate the progression for
    // @return          The stream that was passed in, now modified
    std::ostream& printStateChain(std::ostream& os, const std::uint64_t* order) const {
        // There will be 2^d states in our Markov chain
        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;

        // Represents the state of the system just after test i
        //      where no tests done is considered just after the 0th test
        // Only goes up to N - 1 since we don't care about the state of the system
        //      after the Nth test
        SCCPFloat stateVector[N][numStates];

        stateVector[0][0] = 1.0f; // We begin in state 0 (no colors collected)
        for (std::size_t i = 1; i < numStates; ++i) {
            stateVector[0][i] = 0.0f;
        }

        SCCPFloat E = 1.0f; // Must do first test
        for (std::size_t i = 1; i < N; ++i) {
            std::uint64_t test = order[i - 1];

            // No chance we have no colors after step 1
            stateVector[i][0] = 0.0f;

            // Compute the probability mass that should be in each state
            for (std::size_t state = 1; state < numStates; ++state) {
                stateVector[i][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = (1 << c);
                    std::uint64_t gainedColorBit = colorBit & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = colorBit ^ state;

                    // Pr[got this color on this roll
                    //    OR had this color already, got it on this roll]
                    stateVector[i][state] += 
                    distribution[test][c] * (stateVector[i - 1][stateMinusColor] + stateVector[i - 1][state]);
                }

                if (state != stateFinished) {
                    E += stateVector[i][state];
                }
            }
        }

        os << '\t';
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t biasedState = 0;
            SCCPFloat stateBias = 0.0f;
            for (std::size_t state = 1; state < numStates - 1; ++state) {
                if (stateVector[i][state] > stateBias) {
                    stateBias = stateVector[i][state];
                    biasedState = state;
                }
            }

            os << bitToStateString(biasedState) << '\t';
        }
        os << '\n';

        for (std::uint64_t state : states) {
            os << bitToStateString(state) << '\t';
            for (std::size_t i = 0; i < N; ++i) {
                roundPrint(os, stateVector[i][state]) << '\t';
            }
            os << '\n';
        }
        
        return os;
    }

    // Prints the dice in a permutation as their distributions on [d]
    // @param   os      The stream to print to
    // @param   order   The ordering to print
    // @return          The stream that was passed in, now modified
    std::ostream& printOrderDists(std::ostream& os, const std::uint64_t (&order)[N]) const {
        for (std::size_t c = 0; c < D; ++c) {
            #ifdef PRINT_MARKOV
            os << '\t';
            #endif
            for (std::size_t i = 0; i < N; ++i) {
                roundPrint(os, distribution[order[i]][c]) << '\t';
            }

            os << '\n';
        }

        return os;
    }

    // Brute-force search for an optimal strategy
    void calculateOPT() {
        std::uint64_t currentStrat[N];

        // Array must start sorted for next_permutation to cover all permutations
        for (std::size_t i = 0; i < N; ++i) {
            currentStrat[i] = i;
        }

        std::uint64_t checkedPermutations = 0;
        constexpr std::uint64_t numPermutations = factorial(N);

        // Find optimal ordering by iterating through all strategies
        do {
            if (0 == (++checkedPermutations & OPT_SEARCH_PRINT)) {
                std::cout << "Checked " << checkedPermutations
                          << " permutations out of " << numPermutations << ". ";

                SCCPFloat percentDone = 100.0f * SCCPFloat(checkedPermutations) / numPermutations;
                roundPrint(std::cout, percentDone) << "% complete.                    \r";
            }

            SCCPFloat currentCost = expectedCost(currentStrat);
            if (currentCost < OPT.cost) {
                OPT.cost = currentCost;
                for (std::size_t i = 0; i < N; ++i) {
                    OPT.order[i] = currentStrat[i];
                }
            }
        } while (std::next_permutation(currentStrat, currentStrat + N));
    }

    // Prints information about a given strategy
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printStrategy(std::ostream& os, const NAStrategy& strategy, std::string_view name) const {
        os << name << ":                       \n";
        if (name != "Greedy") {
            printNotGreedy(strategy.order);
        }

        for (std::size_t i = 0; i < N; ++i) {
            os << strategy.order[i] << WHITESPACE;
        }
        os << "\nE[cost(" << name << ", x)]: " << strategy.cost << '\n';

        #ifdef PRINT_DICE_DISTS
        printOrderDists(os << '\n', strategy.order);
        #endif

        #ifdef PRINT_MARKOV
        printStateChain(os << '\n', strategy.order);
        #endif

        return os;
    }

    // Prints information about OPT, assuming it's been found
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printOPT(std::ostream& os) const {
        return printStrategy(os, OPT, "OPT");
    }

    // Generates a greedy strategy with some tests fixed in place at the start
    // @param   order   The array to fill with the greedy strategy
    void greedyFixedTests(std::uint64_t (&order)[N]) {
        bool tested[N]; // Tracks which variables we've used so far
        for (std::size_t test = 0; test < N; ++test) {
            tested[test] = false;
        }

        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;

        // Set up initial state before the first test
        SCCPFloat stateVectors[N + 1][numStates];
        stateVectors[0][0] = 1.0f;
        for (std::size_t state = 1; state < numStates; ++state) {
            stateVectors[0][state] = 0.0f;
        }

        // Calculate the state changes from the fixed tests
        std::size_t turn = 0;
        while (order[turn] != N) {
            stateVectors[turn + 1][0] = 0.0f;

            for (std::size_t state = 1; state < numStates; ++state) {
                stateVectors[turn + 1][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = (1 << c);
                    std::uint64_t gainedColorBit = colorBit & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = colorBit ^ state;

                    // Pr[got this color on this roll
                    //    OR had this color already, got it on this roll]
                    stateVectors[turn + 1][state] +=
                    distribution[order[turn]][c] * (stateVectors[turn][stateMinusColor] + stateVectors[turn][state]);
                }
            }

            tested[order[turn]] = true;
            ++turn;
        }

        for (; turn < N; ++turn) {
            std::uint64_t bestTest = N;
            SCCPFloat bestTestEval = -1.0f;

            for (std::uint64_t candidate = 0; candidate < N; ++candidate) {
                if (tested[candidate]) { continue; }

                SCCPFloat thisTestEval = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t stateJustMissingC = stateFinished ^ (1 << c);

                    thisTestEval +=
                    distribution[candidate][c] * stateVectors[turn][stateJustMissingC];
                }

                if (thisTestEval > bestTestEval) {
                    bestTestEval = thisTestEval;
                    bestTest = candidate;
                }
            }

            order[turn] = bestTest;
            tested[bestTest] = true;

            stateVectors[turn + 1][0] = 0.0f;
            for (std::size_t state = 1; state < numStates; ++state) {
                stateVectors[turn + 1][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = (1 << c);
                    std::uint64_t gainedColorBit = colorBit & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = colorBit ^ state;

                    // Pr[got this color on this roll]
                    stateVectors[turn + 1][state] += distribution[bestTest][c] * stateVectors[turn][stateMinusColor];

                    // Pr[had this color already, got it on this roll]
                    stateVectors[turn + 1][state] += distribution[bestTest][c] * stateVectors[turn][state];
                }
            }
        }
    }

    // Generates a greedy strategy by brute-forcing the first test, then proceeding greedily from there
    void calculateGreedy() {
        std::uint64_t currentGreedy[N];

        for (std::size_t first = 0; first < N - 1; ++first) {
            for (std::size_t second = first + 1; second < N; ++second) {
                currentGreedy[0] = first;
                currentGreedy[1] = second;

                for (std::size_t i = 2; i < N; ++i) {
                    currentGreedy[i] = N;
                }

                greedyFixedTests(currentGreedy);
                SCCPFloat thisStratCost = expectedCost(currentGreedy);

                if (thisStratCost < greedy.cost) {
                    greedy.cost = thisStratCost;
                    for (std::size_t i = 0; i < N; ++i) {
                        greedy.order[i] = currentGreedy[i];
                    }
                }
            }
        }
    }

    // Prints information about Greedy
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printGreedy(std::ostream& os) const {
        return printStrategy(os, greedy, "Greedy");
    }

    // Performs a local search from the given starting point
    // The neighborhood of a permutation is defined by all possible swaps of two tests
    // @param   startingOrder   The permutation to start from
    // @return                  Number of steps to convergence
    std::uint64_t localSearch(std::uint64_t (&startingOrder)[N]) {
        // Copy initial ordering
        localOPT.cost = expectedCost(startingOrder);
        for (std::size_t i = 0; i < N; ++i) {
            localOPT.order[i] = startingOrder[i];
        }

        std::uint64_t stepCount = 0;
        std::size_t bestSwapFrom = N, bestSwapTo = N;
        do {
            bestSwapFrom = bestSwapTo = N;
            SCCPFloat bestCost = localOPT.cost - std::numeric_limits<SCCPFloat>::epsilon(); // Must beat current cost by at least epsilon to be reasonable

            // Check all possible swaps
            for (std::size_t i = 0; i < N - 1; ++i) {
                for (std::size_t j = i + 1; j < N; ++j) {
                    std::swap(localOPT.order[i], localOPT.order[j]);

                    // Compute the new cost of this neighbor
                    SCCPFloat thisNeighborCost = expectedCost(localOPT.order);

                    // Is it better than the previous swaps?
                    if (thisNeighborCost < bestCost) {
                        bestCost = thisNeighborCost;
                        bestSwapFrom = i;
                        bestSwapTo = j;
                    }

                    // Swap back so we can check the rest of the swaps
                    std::swap(localOPT.order[i], localOPT.order[j]);
                }
            }

            // Actually perform the swap found to be best
            if (bestSwapFrom != N) {
                std::swap(localOPT.order[bestSwapFrom], localOPT.order[bestSwapTo]);
                localOPT.cost = bestCost;
            }
        // This will only increment stepCount if (bestSwapFrom != N), and won't affect the loop condition
        } while (bestSwapFrom != N && ++stepCount);

        return stepCount;
    }

    std::uint64_t localSearchFromGreedy() {
        return localSearch(greedy.order);
    }

    std::uint64_t localSearchFromRandom() {
        std::uint64_t startingOrder[N];
        for (std::size_t i = 0; i < N; ++i) { 
            startingOrder[i] = i;
        }

        std::random_device rd;
        std::mt19937 mersenne(rd()); // Mersenne Twister
    
        // Randomly shuffle the tests
        std::shuffle(startingOrder, startingOrder + N, mersenne);

        return localSearch(startingOrder);
    }

    // Prints information about a local optimum
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printLocalOPT(std::ostream& os) const {
        return printStrategy(os, localOPT, "Local OPT");
    }

    // Prints "X" markers above the tests where the order does not behave greedily
    // Should only be called just prior to printing the order
    void printNotGreedy(const std::uint64_t (&order)[N]) const {
        bool tested[N];
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;
        SCCPFloat stateVectors[N + 1][numStates];

        stateVectors[0][0] = 1.0f;
        for (std::size_t state = 1; state < numStates; ++state) {
            stateVectors[0][state] = 0.0f;
        }

        for (std::size_t i = 0; i < D; ++i) {
            std::cout << "__";
            tested[order[i]] = true;

            for (std::size_t state = 0; state < numStates; ++state) {
                stateVectors[i + 1][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = 1 << c;
                    if (0 == (state & colorBit)) { continue; }

                    std::uint64_t stateMinusColor = state ^ colorBit;

                    stateVectors[i + 1][state] +=
                    distribution[order[i]][c] * (stateVectors[i][stateMinusColor] + stateVectors[i][state]);
                }
            }
        }

        for (std::size_t i = D; i < N; ++i) {
            std::uint64_t bestTest = N;
            SCCPFloat bestTestEval = -1.0f;

            for (std::size_t candidate = 0; candidate < N; ++candidate) {
                if (tested[candidate]) { continue; }

                for (std::size_t state = 0; state < numStates; ++state) {
                    stateVectors[i + 1][state] = 0.0f;

                    for (std::size_t c = 0; c < D; ++c) {
                        std::uint64_t colorBit = 1 << c;
                        if (0 == (state & colorBit)) { continue; }

                        std::uint64_t stateMinusColor = state ^ colorBit;

                        stateVectors[i + 1][state] +=
                        distribution[candidate][c] * (stateVectors[i][stateMinusColor] + stateVectors[i][state]);
                    }
                }

                if (stateVectors[i + 1][stateFinished] > bestTestEval) {
                    bestTestEval = stateVectors[i + 1][stateFinished];
                    bestTest = candidate;
                }
            }

            if (bestTest != order[i]) {
                std::cout << "X_";
            } else {
                std::cout << "__";
            }

            tested[order[i]] = true;

            for (std::size_t state = 0; state < numStates; ++state) {
                stateVectors[i + 1][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = 1 << c;
                    if (0 == (state & colorBit)) { continue; }

                    std::uint64_t stateMinusColor = state ^ colorBit;

                    stateVectors[i + 1][state] +=
                    distribution[order[i]][c] * (stateVectors[i][stateMinusColor] + stateVectors[i][state]);
                }
            }
        }

        std::cout << '\n';
    }

    SCCPFloat getOPTCost() const {
        return OPT.cost;
    }

    SCCPFloat getGreedyCost() const {
        return greedy.cost;
    }

    SCCPFloat getLocalOPTCost() const {
        return localOPT.cost;
    }

private:
    SCCPFloat distribution[N][D];
    std::uint64_t states[1 << D];

    bool isHighVariance;

    NAStrategy OPT;
    NAStrategy localOPT;
    NAStrategy greedy;
};

int main() {
    constexpr std::uint64_t ITER_COUNT = 1000000;

    SCCPFloat maxRatio = 0.0f;
    std::uint64_t maxIndex = ITER_COUNT;

    SCCP maxInstance(true);
    SCCP currentInstance(true);

    for (std::size_t i = 1; i <= ITER_COUNT; ++i) {

        currentInstance.calculateGreedy();
        currentInstance.localSearchFromGreedy();

        SCCPFloat greedyLocalRatio = currentInstance.getGreedyCost() / currentInstance.getLocalOPTCost();
        
        currentInstance.localSearchFromRandom();

        SCCPFloat randomLocalRatio = currentInstance.getGreedyCost() / currentInstance.getLocalOPTCost();
        SCCPFloat worseRatio = std::max(greedyLocalRatio, randomLocalRatio);

        if (worseRatio > maxRatio) {
            maxRatio = worseRatio;
            maxInstance = currentInstance;
        }

        if ((i & 127) == 0) {
            std::cout << i << " / " << ITER_COUNT << '\r';
        }

        currentInstance.reset();
    }

    maxInstance.printDistribution(std::cout) << '\n';
    maxInstance.printGreedy(std::cout) << '\n';
    maxInstance.printLocalOPT(std::cout) << '\n';

    std::cout << "Ratio: " << maxRatio << '\n';
    
    return 0;
}