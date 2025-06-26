// @file    naeSimulation.cpp
// @author  Evan Brody
// @brief   Simulation code for the generalized NAE function

#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>

constexpr std::size_t D = 3; // Sides on a die
constexpr std::size_t N = 11; // Number of dice

#define PRINT_MARKOV

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
std::size_t findFirstDiscrepancy(const std::size_t* orderA, const std::size_t* orderB) {
    for (std::size_t i = 0; i < N; ++i) {
        if (orderA[i] != orderB[i]) { return i; }
    }

    return N;
}

std::string colorNumberToString(std::size_t c) {
    switch (c) {
        case 1:
            return "R";
        case 2:
            return "G";
        case 3:
            return "B";
        case 4:
            return "Y";
        case 5:
            return "P";
    }

    return "?";
}

// Class for representing the NAE function and testing strategies for evaluation
class NAE {
    // Represents a nonadaptive strategy
    struct NAStrategy {
        std::size_t order[N];
        NAEfloat cost = N;
    };

public:
    NAE(bool highVariance = true) : highVariance(highVariance) {
        if (highVariance) {
            initDistributionHighVariance();
        } else {
            initDistribution();
        }
    }

    // Resets the state of the simulator
    void reset() {
        OPT.cost = N + 1;
        greedy.cost = N + 1;

        for (std::size_t i = 0; i < N; ++i) {
            OPT.order[i] = N;
            greedy.order[i] = N;
        }

        // Get a fresh distribution
        if (highVariance) {
            initDistributionHighVariance();
        } else {
            initDistribution();
        }
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

                NAEfloat otherProbs[D - 1];
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

                NAEfloat scale = 1.0f - distribution[die][biasedDie];
                std::size_t otherProbsIndex = 0;
                for (std::size_t c = 0; c < D; ++c) {
                    if (c == biasedDie) { continue; }

                    distribution[die][c] = otherProbs[otherProbsIndex] * scale;
                    ++otherProbsIndex;
                }
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

    std::ostream& printStateChain(std::ostream& os, const std::size_t (&order)[N]) const {
        NAEfloat stateVectors[N + 1][D + 2];
        constexpr std::size_t numStates = D + 2;
        constexpr std::size_t stateFinished = D + 1;


        stateVectors[0][0] = 1.0f;
        for (std::size_t state = 1; state < numStates; ++state) {
            stateVectors[0][state] = 0.0f;
        }

        // Handle first turn separately
        stateVectors[1][0] = stateVectors[1][stateFinished] = 0.0f;
        for (std::size_t c = 1; c <= D; ++c) {
            stateVectors[1][c] = distribution[order[0]][c];
        }

        // Handle the rest of the turns
        for (std::size_t turn = 1; turn < N; ++turn) {
            stateVectors[turn + 1][0] = 0.0f;

            NAEfloat totalNotFinished = 0.0f;
            for (std::size_t c = 1; c <= D; ++c) {
                stateVectors[turn + 1][c] = stateVectors[turn][c] * distribution[order[turn]][c];
                totalNotFinished += stateVectors[turn + 1][c];
            }

            stateVectors[turn + 1][stateFinished] = 1.0f - totalNotFinished;
        }

        // Biases
        os << '\t';
        for (std::size_t turn = 1; turn < N; ++turn) {
            std::size_t biasedColor = 1;
            for (std::size_t c = 2; c <= D; ++c) {
                if (stateVectors[turn][c] > stateVectors[turn][biasedColor]) {
                    biasedColor = c;
                }
            }

            os << colorNumberToString(biasedColor) << '\t';
        }
        os << '\n';

        // Print
        for (std::size_t state = 0; state < numStates; ++state) {
            if (1 <= state && state <= D) {
                os << colorNumberToString(state);
            }
            os << '\t';

            // Only go up to N because we don't care about the last turn
            for (std::size_t turn = 0; turn < N; ++turn) {
                roundPrint(os, stateVectors[turn][state]) << '\t';
            }
            os << '\n';
        }
        
        return os;
    }

    /// UTILITY ///

    // Expected cost, limited to n steps at most
    NAEfloat expectedCostTruncated(const std::size_t* order, std::size_t n) const {
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
    NAEfloat expectedCost(const std::size_t* order) const {
        return expectedCostTruncated(order, N);
    }

    /// OPT ///

    // Brute-force search for the optimal nonadaptive strategy
    void calculateOPT() {
        std::size_t currentStrat[N];

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

    std::ostream& printStrategy(std::ostream& os, const NAStrategy& strategy, std::string_view name) const {
        os << name << ":\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << strategy.order[i] << ' ';
        }
        os << "\nE[cost(" << name << ")]: " << strategy.cost << '\n';

        #ifdef PRINT_MARKOV
        printStateChain(os, strategy.order);
        #endif

        return os;
    }

    // Prints information about OPT
    std::ostream& printOPT(std::ostream& os) const {
        return printStrategy(os, OPT, "OPT");
    }

    /// GREEDY ///

    // Generates the standard greedy ordering
    void generateGreedy() {
        std::size_t currentGreedy[N];

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
        std::size_t currentGreedy[N]; // Used to hold candidate strategies
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
    void generateGreedyWithFixedTests(std::size_t* order) const {
        // prAllThis[c] = Pr[only seen color c]
        NAEfloat prAllThis[D];
        for (std::size_t c = 0; c < D; ++c) {
            prAllThis[c] = 1.0f;
        }

        bool tested[N]; // Have we tested variable i ?
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        std::size_t numAdded = 0;

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
            std::size_t bestTest = N;

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
    NAEfloat evalTest(const NAEfloat* prAllThis, std::size_t test) const {
        NAEfloat eval = 0.0f;
        for (std::size_t c = 0; c < D; ++c) {
            eval += prAllThis[c] * distribution[test][c];
        }

        return eval;
    }

    // Alternate metric for the greedy algorithm
    NAEfloat evalTestAlternate(const NAEfloat* prAllThis, std::size_t test) const {
        NAEfloat eval = 0.0f;

        // Find the most likely color to have kept us going
        std::size_t mostLikelyColor = N;
        std::size_t prMostLikelyColor = 0.0f;
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
        return printStrategy(os, greedy, "Greedy");
    }

    void initLocalSearchCostTables(const std::size_t* startingOrder) {
        std::size_t first = startingOrder[0];

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
        std::size_t order[N];
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
    NAEfloat gainFromSwap(const std::size_t* order, std::size_t i, std::size_t j) {
        NAEfloat gain = 0.0f;
        for (std::size_t c = 0; c < D; ++c) {
            // We'll need to divide by pi, then multiply by pj, so we precompute that
            NAEfloat changeFactor = distribution[order[j]][c] / distribution[order[i]][c];
            // oldSum - oldSum * changeFactor = oldSum * (1 - changeFactor)
            gain += sumFromTo[c][i][j] * (1.0f - changeFactor);
        }

        return gain;
    }

    void updateLocalSearchCostTables(const std::size_t* order, std::size_t from, std::size_t to) {
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

    double greedyAdaptiveFromFirstTest(std::size_t first) {
        bool tested[N];
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }
        tested[first] = true;

        return 0.0f;
    }
private:
    bool highVariance;

    NAEfloat distribution[N][D];
    NAStrategy OPT;
    NAStrategy greedy;

    NAEfloat prOnly[D][N];
    NAEfloat sumFromTo[D][N][N]; // Inclusive

    NAEfloat EGreedyAdaptive;
};

int main() {
    NAE nae(false);

    for (std::size_t i = 0; i < 10; ++i) {
        nae.calculateOPT();
        nae.printOPT(std::cout) << '\n';

        nae.generateGreedy();
        nae.printGreedy(std::cout) << '\n';

        nae.reset();

        std::cout << "------------------------------------------------------------------\n";
    }

    return 0;
}