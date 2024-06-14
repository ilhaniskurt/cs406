#include <iostream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <limits>

#define PARTITION 512

double *SkipOrder(double *matrix, int n)
{
    // Initialize arrays
    std::vector<bool> rowVisited(n, false);
    std::vector<int> degs(n, 0);
    std::vector<int> rowPerm(n, 0);
    std::vector<int> colPerm(n, 0);

    // Step 1: Calculate initial degrees of each column (Algorithm 4, lines 5-11)
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            if (matrix[i * n + j] != 0)
            {
                degs[j]++;
            }
        }
    }

    int rowIndex = 0;

    // Step 2: Determine the column and row permutations (Algorithm 4, lines 12-25)
    for (int j = 0; j < n; ++j)
    {
        // Find the column with the minimum degree (Algorithm 4, line 14)
        int curCol = std::distance(degs.begin(), std::min_element(degs.begin(), degs.end()));

        // Mark the column as selected by setting its degree to infinity (Algorithm 4, line 15)
        degs[curCol] = std::numeric_limits<int>::max();
        colPerm[j] = curCol;

        // Update rowPerm and rowVisited (Algorithm 4, lines 17-21)
        for (int i = 0; i < n; ++i)
        {
            if (matrix[i * n + curCol] != 0 && !rowVisited[i])
            {
                rowVisited[i] = true;
                rowPerm[rowIndex++] = i;
            }
        }

        // Update degrees of the remaining columns (Algorithm 4, lines 22-24)
        for (int i = 0; i < n; ++i)
        {
            if (matrix[i * n + curCol] != 0)
            {
                for (int k = 0; k < n; ++k)
                {
                    if (matrix[i * n + k] != 0 && degs[k] != std::numeric_limits<int>::max())
                    {
                        degs[k]--;
                    }
                }
            }
        }
    }

    // Step 3: Create the reordered matrix
    double *reorderedMatrix = new double[n * n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            reorderedMatrix[i * n + j] = matrix[rowPerm[i] * n + colPerm[j]];
        }
    }

    return reorderedMatrix;
}

// double SkipPer(int *crs_ptrs, int *crs_colids, double *crs_values, int *ccs_ptrs, int *ccs_rowids, double *ccs_values, int n, double p, double x[64], double &execution_time)
// {
//     execution_time = omp_get_wtime();
// #pragma omp parallel
//     {
//         double myX[64];
//         std::copy(x, x + 64, myX);
//         double myP = 0, prod;
//         unsigned long long int tn11 = (1ULL << (n - 1)) - 1ULL; // tn11 = 2^(n-1)-1
//         int num_threads = omp_get_num_threads();
//         int thread_id = omp_get_thread_num();
//         unsigned long long int c = (tn11 + num_threads - 1) / num_threads;
//         unsigned long long int myStart = (c * thread_id + 1);
//         tn11 = std::min(tn11, c * (thread_id + 1));
//         unsigned long long int g = myStart, prevG = 0, grdiff, min, max;
//         bool skip;
//         int j, i, ptr;
//         double s;

//         while (g <= tn11)
//         {
//             grdiff = (g ^ (g >> 1)) ^ (prevG ^ (prevG >> 1));
//             prevG = g;
//             for (j = 0; j < n; ++j)
//             {
//                 if ((grdiff >> j) & 1)
//                 {
//                     s = 2.0 * (((g ^ (g >> 1)) >> j) & 1) - 1.0;
//                     for (i = ccs_ptrs[j]; i < ccs_ptrs[j + 1]; ++i)
//                     {
//                         myX[ccs_rowids[i]] += s * ccs_values[i];
//                     }
//                 }
//             }
//             prod = 1.0;
//             for (i = 0; i < n; i++) // değiştirdin test et
//             {
//                 prod *= myX[i];
//             }
//             myP += (2.0 * !(g & 1) - 1.0) * prod;

//             skip = false;
//             max = g;
//             for (i = 0; i < n; i++)
//             {
//                 if (!myX[i])
//                 {
//                     skip = true;
//                     min = UINT64_MAX;
//                     for (ptr = crs_ptrs[i]; ptr < crs_ptrs[i + 1]; ++ptr)
//                     {
//                         j = crs_colids[ptr];
//                         unsigned long long temp = (g < (1ULL << j) ? (1ULL << j) : (g + (1ULL << (j + 1)) - ((g - (1ULL << j)) & ((1ULL << (j + 1)) - 1))));
//                         if (temp < min)
//                         {
//                             min = temp;
//                         }
//                     }
//                     if (max < min)
//                         max = min;
//                 }
//             }
//             if (skip)
//                 g = max;
//             else
//                 ++g;
//         }
// #pragma omp critical
//         p += myP;
//     }
//     execution_time = omp_get_wtime() - execution_time;
//     return ((4 * (n & 1) - 2) * p);
// }

double subSkipPer(int *rptrs, int *columns, double *rvals, int n, int *cptrs, int *rows, double *cvals, double *x, int id)
{
    double myX[64];
    std::copy(x, x + 64, myX);
    double myP = 0, prod;
    unsigned long long int tn11 = (1ULL << (n - 1)) - 1ULL; // tn11 = 2^(n-1)-1
    unsigned long long int c = (tn11 / PARTITION) + 1;
    unsigned long long int myStart = (c * id + 1);
    tn11 = std::min(tn11, c * (id + 1));
    unsigned long long int g = myStart, prevG = 0, grdiff, min, max;
    bool skip;
    int j, i, ptr;
    double s;

    while (g <= tn11)
    {
        grdiff = (g ^ (g >> 1)) ^ (prevG ^ (prevG >> 1));
        prevG = g;
        for (j = 0; j < n; ++j)
        {
            if ((grdiff >> j) & 1)
            {
                s = 2.0 * (((g ^ (g >> 1)) >> j) & 1) - 1.0;
                for (i = cptrs[j]; i < cptrs[j + 1]; ++i)
                {
                    myX[rows[i]] += s * cvals[i];
                }
            }
        }
        prod = 1.0;
        for (i = 0; i < n; i++) // değiştirdin test et
        {
            prod *= myX[i];
        }
        myP += (2.0 * !(g & 1) - 1.0) * prod;
        skip = false;
        max = g;
        for (i = 0; i < n; i++)
        {
            if (!myX[i])
            {
                skip = true;
                min = UINT64_MAX;
                for (ptr = rptrs[i]; ptr < rptrs[i + 1]; ++ptr)
                {
                    j = columns[ptr];
                    unsigned long long temp = (g < (1ULL << j) ? (1ULL << j) : (g + (1ULL << (j + 1)) - ((g - (1ULL << j)) & ((1ULL << (j + 1)) - 1))));
                    if (temp < min)
                    {
                        min = temp;
                    }
                }
                if (max < min)
                    max = min;
            }
        }
        if (skip)
            g = max;
        else
            ++g;
    }
    return myP;
}

double SkipPer(int *crs_ptrs, int *crs_colids, double *crs_values, int *ccs_ptrs, int *ccs_rowids, double *ccs_values, int n, double p, double x[64], double &execution_time)
{
    execution_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1) num_threads(16)
    for (int dyna = 0; dyna < PARTITION; ++dyna)
    {
        double myP = subSkipPer(crs_ptrs, crs_colids, crs_values, n, ccs_ptrs, ccs_rowids, ccs_values, x, dyna);
#pragma omp atomic
        p += myP;
    }
    execution_time = omp_get_wtime() - execution_time;
    return ((4 * (n & 1) - 2) * p);
}
