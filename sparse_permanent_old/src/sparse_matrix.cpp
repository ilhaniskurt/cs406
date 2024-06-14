#include "sparse_matrix.hpp"

#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

std::vector<std::pair<int, int>> generateUniquePairs(int n, int totalPairs, std::mt19937 &gen)
{
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(totalPairs);

    // Generate all possible pairs
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            pairs.emplace_back(i, j);
        }
    }

    // Shuffle the pairs
    std::shuffle(pairs.begin(), pairs.end(), gen);

    // Select the first n^2 * sparsity pairs
    pairs.resize(totalPairs);

    return pairs;
}

// Generate a random sparse square matrix of size n
std::vector<std::vector<float>> generateRandomSparseMatrix(int n, float sparsity)
{
    // Calculate the total number of non-zero elements based on sparsity
    int totalPairs = static_cast<int>(n * n * sparsity);

    std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0));

    // Random generator for floats between 0 and 1 (excluding 0)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rng(0.0f + std::numeric_limits<float>::min(), 1.0f);

    // Generate random row, column, and value using the distributions
    std::vector<std::pair<int, int>> pairs = generateUniquePairs(n, totalPairs, gen);
    for (const std::pair<int, int> &pair : pairs)
        matrix[pair.first][pair.second] = rng(gen);

    return matrix;
}

void displayMatrix(const std::vector<std::vector<float>> &matrix)
{
    for (const std::vector<float> &row : matrix)
    {
        for (const float &value : row)
            std::cout << std::fixed << std::setprecision(6) << value << " ";
        std::cout << std::endl;
    }
}