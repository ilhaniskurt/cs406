#include "sparse_matrix_vector.hpp"

#include <random>
#include <algorithm>
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

    // Sort pairs by first element, then by second
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b)
              { return a.first > b.first || (a.first == b.first && a.second > b.second); });

    return pairs;
}

// Generate a random sparse square matrix of size n
SparseMatrixVector generateRandomSparseMatrix(int n, float sparsity)
{
    // Calculate the total number of non-zero elements based on sparsity
    int totalPairs = static_cast<int>(n * n * sparsity);

    SparseMatrixVector sparseMatrix;
    sparseMatrix.reserve(totalPairs);

    // Random generator for floats between 0 and 1 (excluding 0)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rng(0.0f + std::numeric_limits<float>::min(), 1.0f);

    // Generate random row, column, and value using the distributions
    std::vector<std::pair<int, int>> pairs = generateUniquePairs(n, totalPairs, gen);
    while (!pairs.empty())
    {
        std::pair<int, int> pair = pairs.back();
        float value = rng(gen);
        sparseMatrix.push_back(SparseMatrixEntry(pair, value));
        pairs.pop_back();
    }

    return sparseMatrix;
}

void displaySparseMatrix(SparseMatrixVector &entries, int n)
{
    std::cout << "---------------------------------- Matrix Entries ---------------------------------------" << std::endl
              << std::endl;
    for (auto &entry : entries)
    {
        std::cout << entry << std::endl;
    }
    std::cout << std::endl;

    std::cout << "--------------------------------- Matrix Represantation ---------------------------------" << std::endl
              << std::endl;
    int currentRow = 0, currentCol = 0;
    for (const SparseMatrixEntry &entry : entries)
    {
        if (entry.coordinate.first > currentRow)
        {
            for (; currentRow < entry.coordinate.first; currentRow++)
            {
                for (; currentCol < n; currentCol++)
                {
                    std::cout << "0.000000 ";
                }
                std::cout << std::endl
                          << std::endl;
                currentCol = 0;
            }
        }
        for (; currentCol < entry.coordinate.second; currentCol++)
        {
            std::cout << "0.000000 ";
        }
        std::cout << std::fixed << std::setprecision(6) << entry.value << " ";
        currentCol++;
    }
    for (; currentCol < n; currentCol++)
    {
        std::cout << "0.000000 ";
    }
    std::cout << std::endl
              << std::endl;
    for (; currentRow < n - 1; currentRow++)
    {
        for (int i = 0; i < n; i++)
        {
            std::cout << "0.000000 ";
        }
        std::cout << std::endl
                  << std::endl;
    }
}

double subCalculatePermanent(SparseMatrixVector &entries, int matrix_n, int submatrix_n, int entry_index, int row, int column)
{
    // Base case 1
    if (matrix_n == 1)
        if (entries.empty())
            return 0;
        else
            return entries[entry_index].value;

    // Base case 2
    if (matrix_n == 2)
        if (entries.empty())
            return 0;
        else
        {
            float matrix[2][2];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    matrix[i][j] = 0;
            for (const SparseMatrixEntry &entry : entries)
                matrix[entry.coordinate.first][entry.coordinate.second] = entry.value;
            return static_cast<double>(matrix[0][0] * matrix[1][1]) + static_cast<double>(matrix[0][1] * matrix[1][0]);
        }

    // Recursive Case 1
    if (submatrix_n == 2)
    {
        if (entries.size() == entry_index)
            return 0;
        else
        {
            int terms[4];
        }
    }

    return -1;
}

double calculatePermanent(SparseMatrixVector &entries, int n)
{
    return subCalculatePermanent(entries, n, n, 0, 0, 0);
}
