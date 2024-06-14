#pragma once
#include <vector>

std::vector<std::vector<float>> generateRandomSparseMatrix(int n, float sparsity = 0.1);
void displayMatrix(const std::vector<std::vector<float>> &matrix);