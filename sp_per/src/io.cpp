#include <iostream>
#include <fstream>
#include <cstring>

#include "types.hpp"

void print_matrix(double *matrix, int size)
{
    {
        int count = 0;
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (matrix[i * size + j] != 0)
                {
                    count++;
                }
            }
        }

        std::cout << size << " " << count << std::endl;
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (matrix[i * size + j] != 0)
                {
                    std::cout << i << " " << j << " " << matrix[i * size + j] << std::endl;
                }
            }
        }
    }
}

bool read_matrix(double *&matrix, int &n, int &nonzeros, const char *fname)
{
    std::string filename = fname; // Replace with your file name
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: could not open file " << filename << std::endl;
        exit(1);
    }

    file >> n >> nonzeros;

    matrix = new double[n * n];
    memset(matrix, 0, sizeof(double) * n * n);

    // Check for binary matrix
    bool is_binary = true;

    for (int i = 0; i < nonzeros; ++i)
    {
        int row_id, col_id;
        double nnz_value;
        file >> row_id >> col_id >> nnz_value;
        matrix[(row_id * n) + col_id] = nnz_value;

        if (nnz_value != 1)
        {
            is_binary = false;
        }
    }
    file.close();

    return is_binary;
}