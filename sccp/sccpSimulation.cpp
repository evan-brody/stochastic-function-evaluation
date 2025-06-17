// @file    sccpSimulation.cpp
// @author  Evan Brody
// @brief   Environment for simulating (nonadaptive) strategies for the stochastic coupon collection problem.
//          Assumed unit-cost everywhere

#include <random>
#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <exception>

constexpr std::uint64_t D = 3;      // Number of coupons
constexpr std::uint64_t N = 10;     // Number of tests

// When we're searching for the optimal strategy, we print our progress
//      every OPT_SEARCH_PRINT permutations checked
// Making it a power of 2 minus one allows for fast modulo via bitwise AND
constexpr std::uint64_t OPT_SEARCH_PRINT = (1 << 20) - 1;

// #define DEBUG

typedef long double SCCPFloat;

// Prints a number rounded to 2 decimal places
// @param   os  The stream to print to
// @param   num The number to print
// @return      The stream that was passed in, now modified
std::ostream& roundPrint(std::ostream& os, double num) {
    os << std::round(num * 100.0f) * 0.01f;

    return os;
}

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
    return ((uCount + (uCount >> 3)) & 030707070707) % 63;
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
        unsigned order[N];
        SCCPFloat cost = N + 1;
    };
public:
    SCCP() {
        initDistribution();
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
    SCCPFloat expectedCost(const unsigned* order) const {
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
            unsigned test = order[i - 1];
            const SCCPFloat* testDist = distribution[test];

            stateVector[i][0] = 0.0f; // No chance we have no colors after step 1
            for (std::size_t state = 1; state < numStates; ++state) {
                stateVector[i][state] = 0.0f;

                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t colorBit = (1 << c);
                    std::uint64_t gainedColorBit = colorBit & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = colorBit ^ state;

                    // Pr[got this color on this roll]
                    stateVector[i][state] += testDist[c] * stateVector[i - 1][stateMinusColor];

                    // Pr[had this color already, got it on this roll]
                    stateVector[i][state] += testDist[c] * stateVector[i - 1][state];
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

    // Brute-force search for an optimal strategy
    void calculateOPT() {
        unsigned currentStrat[N];

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
                roundPrint(std::cout, percentDone) << "% complete.\t\t\t\r";
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

    // Prints information about OPT, assuming it's been found
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printOPT(std::ostream& os) const {
        os << "OPT:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << OPT.order[i] << ' ';
        }
        os << "\nE[cost(OPT, x)]: " << OPT.cost;

        return os;
    }

    // Generates the nonadaptive greedy ordering
    void calculateGreedy() {
        unsigned currentGreedy[N];

        // Brute-force to see which test to do first
        for (std::size_t i = 0; i < N; ++i) {
            // Generate the greedy strategy with this first test
            calculateGreedyWithFirstTest(currentGreedy, i);

            // Is it the best so far ?
            SCCPFloat thisCost = expectedCost(currentGreedy);
            if (thisCost < greedy.cost) { // If it's the best so far, copy over its info
                greedy.cost = thisCost;
                for (std::size_t j = 0; j < N; ++j) {
                    greedy.order[j] = currentGreedy[j];
                }
            }
        }
    }

    // Prints information about the generated greedy algorithm
    // @param   os  The stream to print to
    // @return      The stream that was passed in, now modified
    std::ostream& printGreedy(std::ostream& os) const {
        os << "Greedy:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << greedy.order[i] << ' ';
        }
        os << "\nE[cost(Greedy, x)]: " << greedy.cost;

        return os;
    }

    // Calculates what the greedy algorithm would do after the specified first test
    // @param   first   The test to start with
    // @param   order   The initially empty ordering, which will be modified in-place
    void calculateGreedyWithFirstTest(unsigned* order, unsigned first) {
        bool tested[N]; // Tracks which variables we've used so far
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        // First test is fixed, so track that it's been used
        order[0] = first;
        tested[first] = true;

        // There are 2^d possible states, one for each subset of the colors
        constexpr std::uint64_t numStates = 1 << D;

        // stateDist[i][j] = Pr[we're in state j just after the ith test]
        //                      where no tests conducted is considered just after th 0th test
        //                      where order[0] is considered the 1st test
        SCCPFloat stateDist[N + 1][numStates];

        // We always start with no colors with probability 1
        stateDist[0][0] = 1.0f;
        for (std::size_t state = 1; state < numStates; ++state) {
            stateDist[0][state] = 0.0f;
        }

        // Initially clear the distribution for just after the first test
        for (std::size_t state = 0; state < numStates; ++state) {
            stateDist[1][state] = 0.0f;
        }

        // Move the probability mass according to the first test
        for (std::size_t c = 0; c < D; ++c) {
            // The state in which we only have c
            std::uint64_t onlyThisColorState = 1 << c;

            // Pr[get c from first test]
            stateDist[1][onlyThisColorState] = distribution[first][c];
        }

        // Fill the ordering turn by turn
        for (std::size_t turn = 1; turn < N; ++turn) {
            // NOTE: the distribution over the states, at the current point in time,
            //       will be given by stateDist[turn]
            unsigned bestTest = N;
            SCCPFloat bestTestEval = 0.0f; // Evaluation metric we use to compare between possible tests

            // Iterate through candidate tests to find the best
            for (std::size_t candidate = 0; candidate < N; ++candidate) {
                if (tested[candidate]) { continue; } // Skip if used already

                SCCPFloat eval = 0.0f;
                for (std::size_t state = 1; state < numStates; ++state) {
                    SCCPFloat pmExiting = 0.0f; // The probability mass exiting this state

                    // Add in the proportion of probability mass that will move out due to each color
                    for (std::size_t c = 0; c < D; ++c) {
                        if (!((1 << c) & state)) { // If the state needs color c
                            pmExiting += distribution[candidate][c];
                        }
                    }

                    // So far we've just added in proportions, now we need to
                    // scale by the probability mass that's actually in the state
                    pmExiting *= stateDist[turn][state];

                    // Add to our eval
                    eval += pmExiting;
                }

                // Larger eval is better
                if (eval > bestTestEval) {
                    bestTestEval = eval;
                    bestTest = candidate;
                }
            }

            // Every test should have a positive evaluation, so this should never occur
            if (bestTest == N) {
                std::cout << turn << std::endl;
                throw std::runtime_error(
                    "Error: no test found with positive evaluation for greedy algorithm."
                );
            }

            // Add the test into our ordering, and track that we've done it
            order[turn] = bestTest;
            tested[bestTest] = true;

            // Impossible to have no colors after the first test
            stateDist[turn][0] = 0.0f;

            // Move probability mass around the distribution accordingly
            // NOTE: we are wasting a little bit of time by updating the table
            //       after the last test is selected, but this shouldn't matter
            for (std::size_t state = 1; state < numStates; ++state) {
                // Find all the colors this state has
                for (std::size_t c = 0; c < D; ++c) {
                    std::uint64_t gainedColorBit = (1 << c) & state;
                    if (0 == gainedColorBit) { continue; }

                    // The state without color c
                    std::uint64_t stateMinusColor = state ^ gainedColorBit;

                    // Pr[we got c on this test] * Pr[we had everything we do now, except C
                    //                                    OR we had everything we do now]
                    stateDist[turn + 1][state] =
                        distribution[bestTest][c] * (stateDist[turn][stateMinusColor]
                                                   + stateDist[turn][state]);
                }
            }
        }
    }

private:
    SCCPFloat distribution[N][D];
    NAStrategy OPT;
    NAStrategy greedy;
};

int main() {
    for (std::uint64_t _ = 0; _ < 10; ++_) {

        SCCP sccp;

        sccp.printDistribution(std::cout) << '\n';

        sccp.calculateGreedy();
        sccp.printGreedy(std::cout) << '\n';

        sccp.calculateOPT();
        sccp.printOPT(std::cout) << '\n';

    }

    return 0;
}