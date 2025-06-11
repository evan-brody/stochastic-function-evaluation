// @file    naeSimulation.cpp
// @author  Evan Brody
// @brief   Code for testing strategies for evaluating the NAE function

#include <random>
#include <algorithm>
#include <iostream>

#define D 3 // Sides on a die
#define N 10 // Number of dice

// Prints a number rounded to 3 decimal places
std::ostream& roundPrint(std::ostream& os, double num) {
    os << std::round(num * 1000.0f) * 0.001f;

    return os;
}

class NAE {
    // Non-adaptive strategy
    struct NAStrategy {
        unsigned order[N];
        double cost = N;
    };
public:
    NAE() {
        initDistribution();
    }

    ~NAE() {

    }

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

    double expectedCost(unsigned* order) const {
        // prAllThis[c] = Pr[only seen color c]
        double prAllThis[D];

        // Expected cost. Starts at 2 because that's the theoretical minimum
        double E = 2.0f;

        // E[X] = E[X >= 1] + E[X >= 2] + ... + E[X >= N]
        for (std::size_t c = 0; c < D; ++c) {
            // Probability we do the 3rd iteration because we've only seen color c
            prAllThis[c] = distribution[order[0]][c] * distribution[order[1]][c];
        }
        for (std::size_t i = 2; i < N; ++i) {
            for (std::size_t c = 0; c < D; ++c) {
                E += prAllThis[c];
                prAllThis[c] *= distribution[order[i]][c];
            }
        }

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
private:
    double distribution[N][D];
    NAStrategy OPT;
};

int main() {
    NAE nae;
    nae.printDistribution(std::cout) << '\n';
    nae.calculateOPT();
    nae.printOPT(std::cout) << '\n';

    return 0;
}