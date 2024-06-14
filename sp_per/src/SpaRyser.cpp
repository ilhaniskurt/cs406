#include <iostream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <limits>

// Function to count non-zero elements in a column
int countNonZeros(const double *matrix, int n, int col)
{
    int count = 0;
    for (int i = 0; i < n; ++i)
    {
        if (matrix[i * n + col] != 0.0)
        {
            ++count;
        }
    }
    return count;
}

// Function to reorder the columns based on non-zero counts
double *SortOrder(double *matrix, int n)
{
    std::vector<int> colIndices(n);
    for (int i = 0; i < n; ++i)
    {
        colIndices[i] = i;
    }

    // Sort columns by number of non-zero elements
    std::sort(colIndices.begin(), colIndices.end(), [&matrix, n](int a, int b)
              { return countNonZeros(matrix, n, a) < countNonZeros(matrix, n, b); });

    // Allocate memory for the new ordered matrix
    double *orderedMatrix = new double[n * n];

    // Reorder columns in the new matrix
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            orderedMatrix[i * n + j] = matrix[i * n + colIndices[j]];
        }
    }

    return orderedMatrix;
}

double SpaRyser(int *ccs_ptrs, int *ccs_rowids, double *ccs_values, int n, double p, double x[64], double &execution_time)
{
    execution_time = omp_get_wtime();
#pragma omp parallel
    {
        double myX[64];
        std::copy(x, x + 64, myX);
        double myP = 0, prod = 1;
        unsigned long long int tn11 = (1ULL << (n - 1)) - 1ULL; // tn11 = 2^(n-1)-1
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        unsigned long long int c = (tn11 + num_threads - 1) / num_threads;
        unsigned long long int myStart = (c * thread_id + 1);
        tn11 = std::min(tn11, c * (thread_id + 1));
        unsigned long long int g = myStart, prevG = ((myStart - 1) ^ ((myStart - 1) >> 1)), gray, two_to_k;
        int i, j, ptr;
        int zeroNum = 0;
        double s;
        for (i = 0; i < n; ++i)
        {
            if ((prevG >> i) & 1)
            {
                for (ptr = ccs_ptrs[i]; ptr < ccs_ptrs[i + 1]; ++ptr)
                {
                    myX[ccs_rowids[ptr]] += ccs_values[ptr];
                }
            }
        }
        for (i = 0; i < n; ++i)
        {
            if (myX[i])
                prod *= myX[i];
            else
                ++zeroNum;
        }
        for (; g <= tn11; g++)
        {
            gray = g ^ (g >> 1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
            two_to_k = 1;        // two_to_k = 2 raised to the k power (2^k)
            j = 0;
            while (two_to_k < (gray ^ prevG))
            {
                two_to_k <<= 1; // two_to_k is a bitmask to find location of 1
                j++;
            }
            s = (two_to_k & gray) ? +1.0 : -1.0;
            prevG = gray;
            for (i = ccs_ptrs[j]; i < ccs_ptrs[j + 1]; ++i)
            {
                if (!myX[ccs_rowids[i]])
                {
                    --zeroNum;
                    myX[ccs_rowids[i]] += s * ccs_values[i];
                    prod *= myX[ccs_rowids[i]];
                }
                else
                {
                    prod /= myX[ccs_rowids[i]];
                    myX[ccs_rowids[i]] += s * ccs_values[i];
                    if (!myX[ccs_rowids[i]])
                        ++zeroNum;
                    else
                        prod *= myX[ccs_rowids[i]];
                }
            }
            if (!zeroNum)
                myP += ((2.0 * !(g & 1) - 1.0) * prod);
        }
#pragma omp critical
        p += myP;
    }
    execution_time = omp_get_wtime() - execution_time;
    return ((4 * (n & 1) - 2) * p);
}
