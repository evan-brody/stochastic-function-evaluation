// @file    naeSimulation.cpp
// @author  Evan Brody
// @brief   Simulation code for the generalized NAE function

#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>

#define D 7 // Sides on a die
#define N 10 // Number of dice

typedef double NAEfloat;

// Prints a number rounded to 3 decimal places
// @param   os  The stream to print to
// @param   num The number to print
// @return      The stream that was passed in
std::ostream& roundPrint(std::ostream& os, NAEfloat num) {
    os << std::round(num * 1000.0f) * 0.001f;

    return os;
}

// Returns the first test on which A and B disagree
// @param   orderA  The first ordering
// @param   orderB  The second ordering
// @return          Minimum i such that orderA[i] != orderB[i]
unsigned findFirstDiscrepancy(const unsigned* orderA, const unsigned* orderB) {
    for (std::size_t i = 0; i < N; ++i) {
        if (orderA[i] != orderB[i]) { return i; }
    }

    return N;
}

// Class for representing the NAE function and testing strategies for evaluation
class NAE {
    // Represents a nonadaptive strategy
    struct NAStrategy {
        unsigned order[N];
        NAEfloat cost = N;
    };

public:
    NAE() {
        initDistribution();
    }

    // Resets the state of the simulator
    void reset() {
        OPT.cost = N + 1;
        greedy.cost = N + 1;

        for (std::size_t i = 0; i < N; ++i) {
            OPT.order[i] = N;
            greedy.order[i] = N;
        }

        initDistribution(); // Get a fresh distribution
    }

    /// DISTRIBUTION ///

    // Generates a random distribution over the colors for each die
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


    // Prints the distribution in a readable format
    // @param   os  The stream to print to
    // @return      The stream that was passed in
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

    /// UTILITY ///

    // Expected cost, limited to n steps at most
    NAEfloat expectedCostTruncated(const unsigned* order, unsigned n) const {
        assert(n >= 2);

        // prAllThis[c] = Pr[only seen color c]
        NAEfloat prAllThis[D];

        // Expected cost. Starts at 2 because that's the theoretical minimum
        NAEfloat E = 2.0f;

        // E[X] = E[X >= 1] + E[X >= 2] + ... + E[X >= N]
        for (std::size_t c = 0; c < D; ++c) {
            // Probability we do the 3rd iteration because we've only seen color c
            prAllThis[c] = distribution[order[0]][c] * distribution[order[1]][c];
        }
        for (std::size_t i = 2; i < n; ++i) {
            for (std::size_t c = 0; c < D; ++c) {
                E += prAllThis[c];
                prAllThis[c] *= distribution[order[i]][c];
            }
        }

        return E;
    }

    // Calculates the expected cost of a strategy WRT the current distribution
    NAEfloat expectedCost(const unsigned* order) const {
        return expectedCostTruncated(order, N);
    }

    /// OPT ///

    // Brute-force search for the optimal nonadaptive strategy
    void calculateOPT() {
        unsigned currentStrat[N];

        // Array must start sorted for next_permutation to cover all permutations
        for (std::size_t i = 0; i < N; ++i) {
            currentStrat[i] = i;
        }

        // Find optimal ordering by iterating through all strategies
        do {
            NAEfloat currentCost = expectedCost(currentStrat);
            if (currentCost < OPT.cost) {
                OPT.cost = currentCost;
                for (std::size_t i = 0; i < N; ++i) {
                    OPT.order[i] = currentStrat[i];
                }
            }
        } while (std::next_permutation(currentStrat, currentStrat + N));
    }

    // Prints information about OPT
    std::ostream& printOPT(std::ostream& os) const {
        os << "OPT:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << OPT.order[i] << ' ';
        }
        os << "\nE[cost(OPT, x)]: " << OPT.cost;

        return os;
    }

    /// GREEDY ///

    // Generates the standard greedy ordering
    void generateGreedy() {
        unsigned currentGreedy[N];

        // Brute-force to see which test to do first
        for (std::size_t i = 0; i < N; ++i) {
            // Generate the greedy strategy with this first test
            currentGreedy[0] = i;
            for (std::size_t j = 1; j < N; ++j) {
                currentGreedy[j] = N;
            }
            generateGreedyWithFixedTests(currentGreedy);

            // Is it the best so far ?
            NAEfloat thisCost = expectedCost(currentGreedy);
            if (thisCost < greedy.cost) { // If it's the best so far, copy over its info
                greedy.cost = thisCost;
                for (std::size_t j = 0; j < N; ++j) {
                    greedy.order[j] = currentGreedy[j];
                }
            }
        }
    }

    // Performs slightly better but still not optimal
    void generateGreedyAlternate() {
        unsigned currentGreedy[N]; // Used to hold candidate strategies
        bool tested[N]; // tested[i] = have we used up variable i yet ?
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        // Find best ith test
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t candidateTest = 0; candidateTest < N; ++candidateTest) {
                if (tested[candidateTest]) { continue; }

                // Carry over info from previous iterations
                for (std::size_t j = 0; j < i; ++j) {
                    currentGreedy[j] = greedy.order[j];
                }
                for (std::size_t j = i; j < N; ++j) {
                    currentGreedy[j] = N;
                }

                // Test out how well candidateTest does in the ith position
                currentGreedy[i] = candidateTest;
                generateGreedyWithFixedTests(currentGreedy);

                // Is it the best so far ?
                NAEfloat thisCost = expectedCost(currentGreedy);
                if (thisCost < greedy.cost) { // If it's the best so far, copy over its info
                    greedy.cost = thisCost;
                    for (std::size_t j = 0; j < N; ++j) {
                        greedy.order[j] = currentGreedy[j];
                    }
                }
            }

            // Note that we just used up a variable
            tested[greedy.order[i]] = true;
        }
    }

    // Generates a greedy strategy that must begin with a certain sequence of tests
    void generateGreedyWithFixedTests(unsigned* order) const {
        // prAllThis[c] = Pr[only seen color c]
        NAEfloat prAllThis[D];
        for (std::size_t c = 0; c < D; ++c) {
            prAllThis[c] = 1.0f;
        }

        bool tested[N]; // Have we tested variable i ?
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        unsigned numAdded = 0;

        // Get info from predetermined ordering
        std::size_t i = 0;
        while (i < N && order[i] != N) {
            tested[order[i]] = true;
            ++numAdded;

            for (std::size_t c = 0; c < D; ++c) {
                prAllThis[c] *= distribution[order[i]][c];
            }

            ++i;
        }

        // Continually add tests to the strategy until we're finished
        while (numAdded < N) {
            // Search for the next best test
            unsigned bestTest = N;

            // Pr[we perform this test]
            // Lower is better
            NAEfloat bestEval = std::numeric_limits<NAEfloat>::max();
            
            for (std::size_t i = 0; i < N; ++i) {
                if (tested[i]) { continue; }

                NAEfloat thisTestEval = evalTest(prAllThis, i);

                if (thisTestEval < bestEval) {
                    bestEval = thisTestEval;
                    bestTest = i;
                }
            }

            order[numAdded++] = bestTest;
            tested[bestTest] = true;

            for (std::size_t c = 0; c < D; ++c) {
                prAllThis[c] *= distribution[bestTest][c];
            }
        }
    }

    // Metric that the greedy algorithm uses
    NAEfloat evalTest(const NAEfloat* prAllThis, unsigned test) const {
        NAEfloat eval = 0.0f;
        for (std::size_t c = 0; c < D; ++c) {
            eval += prAllThis[c] * distribution[test][c];
        }

        return eval;
    }
    // Alternate metric for the greedy algorithm
    NAEfloat evalTestAlternate(const NAEfloat* prAllThis, unsigned test) const {
        NAEfloat eval = 0.0f;

        // Find the most likely color to have kept us going
        unsigned mostLikelyColor = N;
        unsigned prMostLikelyColor = 0.0f;
        for (std::size_t c = 0; c < D; ++c) {
            if (prAllThis[c] > prMostLikelyColor) {
                prMostLikelyColor = prAllThis[c];
                mostLikelyColor = c;
            }
        }

        // Take dot product with the gradient
        for (std::size_t c = 0; c < D; ++c) {
            if (c == mostLikelyColor) { continue; }

            eval += (prMostLikelyColor - prAllThis[c]) * distribution[test][c];
        }

        // Negate for proper comparison
        return -eval;
    }

    // Prints information about the generated greedy strategy
    std::ostream& printGreedy(std::ostream& os) const {
        os << "Greedy:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << greedy.order[i] << ' ';
        }
        os << "\nE[cost(Greedy, x)]: " << greedy.cost;

        return os;
    }

    void initLocalSearchCostTables(const unsigned* startingOrder) {
        unsigned first = startingOrder[0];

        // prOnly[c][i] = Pr[only seen color c after rolling (i + 1)th die]
        for (std::size_t c = 0; c < D; ++c) {
            prOnly[c][0] = distribution[first][c];

            for (std::size_t turn = 1; turn < N; ++turn) {
                // Pr[only seen color c after (i + 1)th die]
                // = Pr[only seen color c after ith die] * Pr[(i + 1)th die = c]
                prOnly[c][turn] = prOnly[c][turn - 1] * distribution[startingOrder[turn]][c];
            }
        }

        // Initialize partial sums to 0
        for (std::size_t c = 0; c < D; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < N; ++j) {
                    sumFromTo[c][i][j] = 0.0f;
                }
            }
        }

        // Fill in partial sums
        for (std::size_t c = 0; c < D; ++c) {
            for (std::size_t i = 0; i < N; ++i) {
                sumFromTo[c][i][i] = prOnly[c][i];

                for (std::size_t j = i + 1; j < N; ++j) {
                    sumFromTo[c][i][j] = sumFromTo[c][i][j - 1] + prOnly[c][j];
                }
            }
        }
    }

    std::uint64_t localSearch() {
        // Start from a random ordering
        unsigned order[N];
        for (std::size_t i = 0; i < N; ++i) {
            order[i] = i;
        }

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(order, order + N, g);

        initLocalSearchCostTables(order);

        NAEfloat bestSwapGain = 0.0f;
        std::size_t bestSwapFrom = N;
        std::size_t bestSwapTo = N;
        std::uint64_t stepCount = 0;

        do {
            // Reset at the start of each iteration
            bestSwapGain = 0.0f;
            bestSwapTo = bestSwapFrom = N;

            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t j = i + 1; j < N; ++j) {
                    NAEfloat thisSwapGain = gainFromSwap(order,i, j);
                    if (thisSwapGain > bestSwapGain) {
                        bestSwapGain = thisSwapGain;
                        bestSwapFrom = i;
                        bestSwapTo = j;
                    }
                }
            }

            if (bestSwapFrom != N) {
                updateLocalSearchCostTables(order, bestSwapFrom, bestSwapTo);
                std::swap(order[bestSwapFrom], order[bestSwapTo]);
            }

            if (stepCount % 10000 == 0) {
                std::cout << "Completed " << stepCount << " steps.\n";
            }
        // Only increment stepCount if we're moving on to the next iteration
        } while (bestSwapFrom != N && ++stepCount);

        return stepCount;
    }

    // Assumes i < j
    NAEfloat gainFromSwap(const unsigned* order, std::size_t i, std::size_t j) {
        NAEfloat gain = 0.0f;
        for (std::size_t c = 0; c < D; ++c) {
            // We'll need to divide by pi, then multiply by pj, so we precompute that
            NAEfloat changeFactor = distribution[order[j]][c] / distribution[order[i]][c];
            // oldSum - oldSum * changeFactor = oldSum * (1 - changeFactor)
            gain += sumFromTo[c][i][j] * (1.0f - changeFactor);
        }

        return gain;
    }

    void updateLocalSearchCostTables(const unsigned* order, std::size_t from, std::size_t to) {
        for (std::size_t c = 0; c < D; ++c) {
            NAEfloat changeFactor = distribution[order[from]][c] / distribution[order[to]][c];
            for (std::size_t i = from; i < to; ++i) {
                prOnly[c][i] *= changeFactor;

                for (std::size_t j = i; j < to; ++j) {
                    sumFromTo[c][i][j] *= changeFactor;
                }
            }
        }
    }

    void greedyAdaptive() {

        for (std::size_t i = 0; i < N; ++i) {

        }
    }

    double greedyAdaptiveFromFirstTest(unsigned first) {
        bool tested[N];
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }
        tested[first] = true;

        return 0.0f;
    }
private:
    NAEfloat distribution[N][D];
    NAStrategy OPT;
    NAStrategy greedy;

    NAEfloat prOnly[D][N];
    NAEfloat sumFromTo[D][N][N]; // Inclusive

    NAEfloat EGreedyAdaptive;
};

int main() {
    NAE nae;

    for (std::size_t i = 0; i < 1; ++i) {
        nae.calculateOPT();
        nae.printOPT(std::cout) << '\n';

        nae.generateGreedy();
        nae.printGreedy(std::cout) << '\n';

        // std::uint64_t stepCount = nae.localSearch();
        // std::cout << "Steps: " << stepCount << std::endl;

        nae.reset();
    }

    return 0;
}