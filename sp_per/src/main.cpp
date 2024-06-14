#include <iostream>
#include <cstring>
#include "omp.h"

#include "io.hpp"
#include "form.hpp"
#include "SkipPer.hpp"
#include "SpaRyser.hpp"

#define OMP_NUM_THREADS 16

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // Set the number of threads
    omp_set_num_threads(OMP_NUM_THREADS);

    double *unordered_matrix = nullptr;
    int n, nonzeros;

    bool is_binary = read_matrix(unordered_matrix, n, nonzeros, argv[1]);

    // float sparsity = (float)nonzeros / (n * n);

    // std::cout << "Is Binary: " << is_binary << std::endl;

    double *matrix;

    // is_binary = true;

    if (is_binary)
    {
        matrix = SkipOrder(unordered_matrix, n);
    }
    else
    {
        matrix = SortOrder(unordered_matrix, n);
    }

    delete[] unordered_matrix;

    int *crs_ptrs;
    int *crs_colids;
    double *crs_values;
    int *ccs_ptrs;
    int *ccs_rowids;
    double *ccs_values;

    sparseMatrix *data = createSparseMatrix(matrix, n, nonzeros);
    formCompressed(data, crs_ptrs, crs_colids, crs_values, ccs_ptrs, ccs_rowids, ccs_values, n, nonzeros);

    // Preprocessing
    double p = 1.0;
    double x[64];
    {
        int i = 0;
        int j = 0;
        double rowSum;
        for (i = 0; i < n; ++i)
        {
            rowSum = 0;
            for (j = 0; j < n; ++j)
            {
                rowSum += matrix[i * n + j];
            }
            x[i] = matrix[i * n + n - 1] - rowSum / 2.0;
            p *= x[i];
        }
    }

    double permanent;
    double execution_time;

    // is_binary = false;

    if (is_binary)
    {
        permanent = SkipPer(crs_ptrs, crs_colids, crs_values, ccs_ptrs, ccs_rowids, ccs_values, n, p, x, execution_time);
    }
    else
    {
        permanent = SpaRyser(ccs_ptrs, ccs_rowids, ccs_values, n, p, x, execution_time);
    }

    std::cout << permanent << " " << execution_time << std::endl;

    // Teardown
    delete[] matrix;
    delete[] data;
    delete crs_ptrs;
    delete crs_colids;
    delete crs_values;
    delete ccs_ptrs;
    delete ccs_rowids;
    delete ccs_values;
    return 0;
}