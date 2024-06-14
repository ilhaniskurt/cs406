#pragma once
#include <vector>
#include <iostream>

// Struct to represent a non-zero element in the sparse matrix
struct SparseMatrixEntry
{
    std::pair<int, int> coordinate;
    float value;

    SparseMatrixEntry(std::pair<int, int> co, float val) : coordinate(co), value(val) {}

    friend std::ostream &operator<<(std::ostream &os, const SparseMatrixEntry &matrix)
    {
        return os << "( Row: " << matrix.coordinate.first << ", Column: " << matrix.coordinate.second << ", Value: " << matrix.value << " )";
    }
};

using SparseMatrixVector = std::vector<SparseMatrixEntry>;

SparseMatrixVector generateRandomSparseMatrix(int n, float sparsity = 0.1);
void displaySparseMatrix(SparseMatrixVector &entries, int n);
double calculatePermanent(SparseMatrixVector &entries, int n);