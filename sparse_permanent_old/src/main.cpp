#include "sparse_matrix.hpp"
#define MATRIX_SIZE 4

int main()
{
    std::vector<std::vector<float>> matrix = generateRandomSparseMatrix(MATRIX_SIZE, 1);
    displayMatrix(matrix);
    return 0;
}