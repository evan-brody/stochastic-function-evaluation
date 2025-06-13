// @file    naeSimulation.hpp
// @author  Evan Brody
// @brief   Outlines the simulation class and several helper functions

#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>

#define D 7 // Sides on a die
#define N 10 // Number of dice

// Prints a number rounded to 3 decimal places
// @param   os  The stream to print to
// @param   num The number to print
// @return      The stream that was passed in
std::ostream& roundPrint(std::ostream& os, double num);

// Returns the first test on which A and B disagree
// @param   orderA  The first ordering
// @param   orderB  The second ordering
// @return          Minimum i such that orderA[i] != orderB[i]
unsigned findFirstDiscrepancy(const unsigned* orderA, const unsigned* orderB);

// Class for representing the NAE function and testing strategies for evaluation
class NAE {
    // Represents a nonadaptive strategy
    struct NAStrategy {
        unsigned order[N];
        double cost = N;
    };

public:
    NAE();

    // Resets the state of the simulator
    void reset();

    /// DISTRIBUTION ///

    // Generates a random distribution over the colors for each die
    void initDistribution();
    // Prints the distribution in a readable format
    // @param   os  The stream to print to
    // @return      The stream that was passed in
    std::ostream& printDistribution(std::ostream& os) const;

    /// UTILITY ///

    // Expected cost, limited to n steps at most
    double expectedCostTruncated(const unsigned* order, unsigned n) const;
    // Calculates the expected cost of a strategy WRT the current distribution
    double expectedCost(const unsigned* order) const;

    /// OPT ///

    // Brute-force search for the optimal nonadaptive strategy
    void calculateOPT();
    // Prints information about OPT
    std::ostream& printOPT(std::ostream& os) const;

    /// GREEDY ///

    // Generates the standard greedy ordering
    void generateGreedy();
    // Performs slightly better but still not optimal
    void generateGreedyAlternate();
    // Generates a greedy strategy that must begin with a certain sequence of tests
    void generateGreedyWithFixedTests(unsigned* order) const;

    // Metric that the greedy algorithm uses
    double evalTest(const double* prAllThis, unsigned test);
    // Alternate metric for the greedy algorithm
    double evalTestAlternate(const double* prAllThis, unsigned test);
    // Prints information about the generated greedy strategy
    std::ostream& printGreedy(std::ostream& os) const;

private:
    double distribution[N][D];
    NAStrategy OPT;
    NAStrategy greedy;
};