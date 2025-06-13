// @file    sccpSimulation.cpp
// @author  Evan Brody

#include <random>
#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>

#define D 3
#define N 10

// #define DEBUG

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

class SCCP {
    // Represents a nonadaptive strategy
    struct NAStrategy {
        unsigned order[N];
        double cost = N + 1;
    };
public:
    SCCP() {
        initDistribution();
    }

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

    double expectedCost(const unsigned* order) const {
        // There will be 2^d states in our Markov chain
        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;
        double stateVector[N + 1][numStates]; // Represents the state of the system just after test i
        stateVector[0][0] = 1.0f; // We begin in state 0 (no colors collected)
        for (std::size_t i = 1; i < numStates; ++i) {
            stateVector[0][i] = 0.0f;
        }

        double E = 0.0f;
        for (std::size_t i = 1; i < N + 1; ++i) {
            unsigned test = order[i - 1];
            const double* testDist = distribution[test];

            stateVector[i][0] = 0.0f; // No chance we have no colors after step 1
            for (std::size_t state = 1; state < numStates; ++state) {
                stateVector[i][state] = 0.0f;

                for (std::size_t color = 1; color <= (1 << (D - 1)); color <<= 1) {
                    std::uint64_t gainedColorBit = color & state;
                    if (0 == gainedColorBit) { continue; }

                    std::uint64_t stateMinusColor = color ^ state;
                    std::uint64_t gainedColor = log2(gainedColorBit);

                    // Pr[got this color on this roll]
                    stateVector[i][state] += testDist[gainedColor] * stateVector[i - 1][stateMinusColor];

                    // Pr[had this color already, got it on this roll]
                    stateVector[i][state] += testDist[gainedColor] * stateVector[i - 1][state];
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

    void calculateOPT() {
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

    std::ostream& printOPT(std::ostream& os) const {
        os << "OPT:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << OPT.order[i] << ' ';
        }
        os << "\nE[cost(OPT, x)]: " << OPT.cost;

        return os;
    }

    void calculateGreedy() {
        unsigned currentGreedy[N];

        // Brute-force to see which test to do first
        for (std::size_t i = 0; i < N; ++i) {
            // Generate the greedy strategy with this first test
            calculateGreedyWithFirstTest(currentGreedy, i);

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

    std::ostream& printGreedy(std::ostream& os) const {
        os << "Greedy:\n";
        for (std::size_t i = 0; i < N; ++i) {
            os << greedy.order[i] << ' ';
        }
        os << "\nE[cost(Greedy, x)]: " << greedy.cost;

        return os;
    }

    double evalTest(double** stateVector, unsigned candidateTest, std::size_t turn) {
        constexpr std::uint64_t numStates = 1 << D;
        constexpr std::uint64_t stateFinished = numStates - 1;

        const double* candDist = distribution[candidateTest];
        double eval = 0.0f;

        for (std::size_t state = 1; state < numStates; ++state) {
            for (std::size_t color = 1; color <= (1 << (D - 1)); color <<= 1) {
                std::uint64_t gainedColorBit = color & state;
                if (0 == gainedColorBit) { continue; }

                std::uint64_t stateMinusColor = color ^ state;
                std::uint64_t gainedColor = log2(gainedColorBit);

                // Pr[got this color on this roll]
                stateVector[turn][state] += 1; // candDist[gainedColor]; // * stateVector[turn - 1][stateMinusColor];

                // Pr[had this color already, got it on this roll]
                stateVector[turn][state] += candDist[gainedColor] * stateVector[turn - 1][state];
            }

            if (state != stateFinished) {
                eval += stateVector[turn][state];
            }
        }

        return eval;
    }

    void calculateGreedyWithFirstTest(unsigned* order, unsigned first) {
        bool tested[N];
        for (std::size_t i = 0; i < N; ++i) {
            tested[i] = false;
        }

        order[0] = first;
        tested[first] = true;

        double prNotSeen[D];
        for (std::size_t c = 0; c < D; ++c) {
            prNotSeen[c] = 1.0f - distribution[first][c];
        }

        for (std::size_t turn = 1; turn < N; ++turn) {
            unsigned bestTest = N;
            double bestTestEval = std::numeric_limits<double>::max();
            for (std::size_t candidate = 0; candidate < N; ++candidate) {
                if (tested[candidate]) { continue; }

                double eval = 0.0f;
                for (std::size_t c = 0; c < D; ++c) {
                    eval += prNotSeen[c] * (1.0f - distribution[candidate][c]);
                }

                if (eval < bestTestEval) {
                    bestTestEval = eval;
                    bestTest = candidate;
                }
            }

            order[turn] = bestTest;
            tested[bestTest] = true;
        }
    }


private:
    double distribution[N][D];
    NAStrategy OPT;
    NAStrategy greedy;
};

int main() {
    SCCP sccp;

    sccp.printDistribution(std::cout) << '\n';

    sccp.calculateOPT();
    sccp.printOPT(std::cout) << '\n';

    sccp.calculateGreedy();
    sccp.printGreedy(std::cout) << '\n';

    return 0;
}