#include "types.hpp"

#include <cstring>
#include <cstdlib>
#include <iostream>

void formCompressed(sparseMatrix *X,
                    int *&crs_ptrs,
                    int *&crs_colids,
                    double *&crs_values,
                    int *&ccs_ptrs,
                    int *&ccs_rowids,
                    double *&ccs_values,
                    int n,
                    int nonzeros)
{
    crs_ptrs = (int *)malloc((n + 1) * sizeof(int));
    crs_colids = (int *)malloc(nonzeros * sizeof(int));
    crs_values = (double *)malloc(nonzeros * sizeof(double));
    ccs_ptrs = (int *)malloc((n + 1) * sizeof(int));
    ccs_rowids = (int *)malloc(nonzeros * sizeof(int));
    ccs_values = (double *)malloc(nonzeros * sizeof(double));
    // fill the crs_ptrs array
    memset(crs_ptrs, 0, (n + 1) * sizeof(int));
    for (int i = 0; i < nonzeros; i++)
    {
        int rowid = X[i].i;
        if (rowid < 0 || rowid >= n)
        {
            printf("problem in X, quitting - %d\n", rowid);
            exit(1);
        }
        crs_ptrs[rowid + 1]++;
    }

    // now we have cumulative ordering of crs_ptrs.
    for (int i = 1; i <= n; i++)
    {
        crs_ptrs[i] += crs_ptrs[i - 1];
    }
    // printf("This number should be equal to the number of nonzeros %d\n", crs_ptrs[n]);

    // we set crs_colids such that for each element, it holds the related column of that element
    for (int i = 0; i < nonzeros; i++)
    {
        int rowid = X[i].i;
        int index = crs_ptrs[rowid];

        crs_colids[index] = X[i].j;
        crs_values[index] = X[i].val;

        crs_ptrs[rowid] = crs_ptrs[rowid] + 1;
    }

    for (int i = n; i > 0; i--)
    {
        crs_ptrs[i] = crs_ptrs[i - 1];
    }
    crs_ptrs[0] = 0;
    // printf("This number should be equal to the number of nonzeros %d\n", crs_ptrs[n]);

    // fill the ccs_ptrs array
    memset(ccs_ptrs, 0, (n + 1) * sizeof(int));
    for (int i = 0; i < nonzeros; i++)
    {
        int colid = X[i].j;
        if (colid < 0 || colid >= n)
        {
            printf("problem in X, quitting - %d\n", colid);
            exit(1);
        }
        ccs_ptrs[colid + 1]++;
    }

    // now we have cumulative ordering of ccs_ptrs.
    for (int i = 1; i <= n; i++)
    {
        ccs_ptrs[i] += ccs_ptrs[i - 1];
    }
    // printf("This number should be equal to the number of nonzeros %d\n", ccs_ptrs[n]);

    // we set ccs_rowids such that for each element, it holds the related row of that element
    for (int i = 0; i < nonzeros; i++)
    {
        int colid = X[i].j;
        int index = ccs_ptrs[colid];

        ccs_rowids[index] = X[i].i;
        ccs_values[index] = X[i].val;

        ccs_ptrs[colid] = ccs_ptrs[colid] + 1;
    }

    for (int i = n; i > 0; i--)
    {
        ccs_ptrs[i] = ccs_ptrs[i - 1];
    }
    ccs_ptrs[0] = 0;
    // printf("This number should be equal to the number of nonzeros %d\n", ccs_ptrs[n]);
}

sparseMatrix *createSparseMatrix(double *matrix, int n, int nonZeroCount)
{
    sparseMatrix *data = (sparseMatrix *)malloc(nonZeroCount * sizeof(sparseMatrix));

    int count = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double val = matrix[i * n + j];
            if (val != 0)
            {
                data[count].i = i;
                data[count].j = j;
                data[count].val = val;
                count++;
            }
        }
    }

    return data;
}