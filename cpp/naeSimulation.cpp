// @file    naeSimulation.cpp
// @author  Evan Brody
// @brief   Code for testing strategies for evaluating the NAE function
//          Documentation is in naeSimulation.hpp

#include "naeSimulation.hpp"

std::ostream& roundPrint(std::ostream& os, double num) {
    os << std::round(num * 1000.0f) * 0.001f;

    return os;
}

unsigned findFirstDiscrepancy(const unsigned* orderA, const unsigned* orderB) {
    for (std::size_t i = 0; i < N; ++i) {
        if (orderA[i] != orderB[i]) { return i; }
    }

    return N;
}


NAE::NAE() {
    initDistribution();
}

void NAE::reset() {
    OPT.cost = N + 1;
    greedy.cost = N + 1;

    for (std::size_t i = 0; i < N; ++i) {
        OPT.order[i] = N;
        greedy.order[i] = N;
    }

    initDistribution(); // Get a fresh distribution
}

void NAE::initDistribution() {
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

std::ostream& NAE::printDistribution(std::ostream& os) const {
    // Rows are dice, columns are colors
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < D; ++c) {
            roundPrint(os, distribution[i][c]) << '\t';
        }
        os << '\n';
    }

    return os;
}

double NAE::expectedCostTruncated(const unsigned* order, unsigned n) const {
    assert(n >= 2);

    // prAllThis[c] = Pr[only seen color c]
    double prAllThis[D];

    // Expected cost. Starts at 2 because that's the theoretical minimum
    double E = 2.0f;

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

double NAE::expectedCost(const unsigned* order) const {
    return expectedCostTruncated(order, N);
}

// Brute-force search for the optimal nonadaptive strategy
void NAE::calculateOPT() {
    unsigned currentStrat[N];

    // Array must start sorted for next_permutation to cover all permutations
    for (std::size_t i = 0; i < N; ++i) {
        currentStrat[i] = i;
    }

    // Find optimal ordering by iterating through all strategies
    do {
        double currentCost = expectedCost(currentStrat);
        if (currentCost < OPT.cost) {
            OPT.cost = currentCost;
            for (std::size_t i = 0; i < N; ++i) {
                OPT.order[i] = currentStrat[i];
            }
        }
    } while (std::next_permutation(currentStrat, currentStrat + N));
}

std::ostream& NAE::printOPT(std::ostream& os) const {
    os << "OPT:\n";
    for (std::size_t i = 0; i < N; ++i) {
        os << OPT.order[i] << ' ';
    }
    os << "\nE[cost(OPT, x)]: " << OPT.cost;

    return os;
}

// Generates the standard greedy ordering
void NAE::generateGreedy() {
    unsigned currentGreedy[N];

    // Brute-force to see which test to do first
    for (std::size_t i = 0; i < N; ++i) {
        // Generate the greedy strategy with this first test
        currentGreedy[0] = i;
        generateGreedyWithFixedTests(currentGreedy);

        // Is it the best so far ?
        double thisCost = expectedCost(currentGreedy);
        if (thisCost < greedy.cost) { // If it's the best so far, copy over its info
            greedy.cost = thisCost;
            for (std::size_t j = 0; j < N; ++j) {
                greedy.order[j] = currentGreedy[j];
            }
        }
    }
}

void NAE::generateGreedyAlternate() {
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
            double thisCost = expectedCost(currentGreedy);
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

void NAE::generateGreedyWithFixedTests(unsigned* order) const {
    // prAllThis[c] = Pr[only seen color c]
    double prAllThis[D];
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
        double bestEval = std::numeric_limits<double>::max();
        
        for (std::size_t i = 0; i < N; ++i) {
            if (tested[i]) { continue; }

            double thisTestEval = evalTest(prAllThis, i);

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

// Helper function for below
double NAE::evalTest(const double* prAllThis, unsigned test) {
    double eval = 0.0f;
    for (std::size_t c = 0; c < D; ++c) {
        eval += prAllThis[c] * distribution[test][c];
    }

    return eval;
}

double NAE::evalTestAlternate(const double* prAllThis, unsigned test) {
    double eval = 0.0f;

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

std::ostream& NAE::printGreedy(std::ostream& os) const {
    os << "Greedy:\n";
    for (std::size_t i = 0; i < N; ++i) {
        os << greedy.order[i] << ' ';
    }
    os << "\nE[cost(Greedy, x)]: " << greedy.cost;

    return os;
}

int main() {
    NAE nae;

    for (std::size_t i = 0; i < 10; ++i) {
        nae.calculateOPT();
        nae.printOPT(std::cout) << '\n';

        nae.generateGreedy();
        nae.printGreedy(std::cout) << '\n';

        nae.reset();
    }

    return 0;
}