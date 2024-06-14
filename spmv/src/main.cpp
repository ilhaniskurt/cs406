#include <iostream>
#include "omp.h"
#include <immintrin.h> // AVX intrinsics
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

using namespace std;

typedef struct sparseMatrix
{
    int i;
    int j;
    double val;
} sparseMatrix;

sparseMatrix *X;
int NumCols, NumRows;
int NumElements;

int loadDataSparse(char *fileName, sparseMatrix **data, int *size1, int *size2)
{
    FILE *myfile;
    if ((myfile = fopen(fileName, "r")) == NULL)
    {
        printf("Error: file cannot be found\n");
        exit(1);
    }

    int s1, s2, numel, result;
    if ((result = fscanf(myfile, "%d %d %d", &s1, &s2, &numel)) <= 0)
    {
        printf("error while reading file %d\n", result);
        exit(1);
    }

    printf("The number of rows, columns, and nonzeros are %d, %d, and %d respectively\n", s1, s2, numel);

    *data = (sparseMatrix *)malloc(numel * sizeof(sparseMatrix));
    for (int i = 0; i < numel; i++)
    {
        int tempi, tempj;
        double tempval;
        if ((result = fscanf(myfile, "%d %d %lf", &tempi, &tempj, &tempval)) <= 0)
        {
            printf("error while reading file - %d\n", result);
            exit(1);
        }

        (*data)[i].i = tempi - 1; // INDEXING STARTS AT 1 in the FILE but WE NEED 0 based index
        (*data)[i].j = tempj - 1; // INDEXING STARTS AT 1 in the FILE but WE NEED 0 based index
        (*data)[i].val = tempval;
    }

    fclose(myfile);

    *size1 = s1;
    *size2 = s2;

    return numel;
}

void iterative_spmv_seq(int *crs_ptrs, int *crs_colids, double *crs_vals, double *in, int no_iterations, double *&out, double *&temp)
{
    printf("sequential spmv starts\n");
    // copy the input to the output; we don't want to change the contents of the input
    for (int i = 0; i < NumRows; i++)
    {
        out[i] = in[i];
    }

    double start = omp_get_wtime();
    // iterative SpMV - take matrix power
    for (int iter = 0; iter < no_iterations; iter++)
    {
        for (int i = 0; i < NumRows; i++)
        {
            temp[i] = 0;
            for (int p = crs_ptrs[i]; p < crs_ptrs[i + 1]; p++)
            {
                temp[i] += crs_vals[p] * out[crs_colids[p]];
            }
        }

        // preparing for the next iteration
        double *t = out;
        out = temp;
        temp = t;
    }
    double end = omp_get_wtime();
    printf("sequential spmv ends in %f seconds\n", end - start);
}

inline double _mm256_reduce_add_pd(__m256d vec)
{
    __m256d temp = _mm256_hadd_pd(vec, vec); // Horizontal add pairs of elements
    __m128d sum_high = _mm256_extractf128_pd(temp, 1);
    __m128d result = _mm_add_pd(_mm256_castpd256_pd128(temp), sum_high); // Final reduction to scalar
    return ((double *)&result)[0];                                       // Extract scalar result
}

void iterative_spmv_par(int *crs_ptrs, int *crs_colids, double *crs_vals, double *in, int no_iterations, double *&out, double *&temp)
{
    // copy the input to the output; we don't want to change the contents of the input
    for (int i = 0; i < NumRows; i++)
    {
        out[i] = in[i];
    }

    double start = omp_get_wtime();

    // AFTER ALL THAT WORK DEALING WITH SEG FAULTS IT TURNS OUT THIS IS SLOWER
    // FUUUUUUUUUUUUUCC GIVE MY 6 HOURS BACK
    // AFTER 6 MORE HOURS NOTHING CHANGED
    //   for (int iter = 0; iter < no_iterations; iter++)
    //   {
    // #pragma omp parallel for default(none) shared(crs_ptrs, crs_colids, crs_vals, out, temp, NumRows)
    //     for (int i = 0; i < NumRows; i++)
    //     {
    //       __m256d sum = _mm256_setzero_pd(); // Initialize sum vector to zero
    //       // Process full blocks of 4 elements
    //       for (int p = crs_ptrs[i]; p + 3 < crs_ptrs[i + 1]; p += 4)
    //       {
    //         __m256d vals = _mm256_load_pd(&crs_vals[p]);
    //         __m256d vec = _mm256_set_pd(out[crs_colids[p + 3]], out[crs_colids[p + 2]], out[crs_colids[p + 1]], out[crs_colids[p]]);
    //         sum = _mm256_fmadd_pd(vals, vec, sum);
    //       }
    //       // Reduce vector sum and add scalar sum
    //       temp[i] = _mm256_reduce_add_pd(sum);
    //     }
    // // preparing for the next iteration
    // #pragma omp single
    //     {
    //       double *t = out;
    //       out = temp;
    //       temp = t;
    //     }
    //   }

    // iterative SpMV - take matrix power
    double *t;
    for (int iter = 0; iter < no_iterations; iter++)
    {
#pragma omp parallel for simd default(none) shared(crs_ptrs, crs_colids, crs_vals, out, temp, NumRows)
        for (int i = 0; i < NumRows; i++)
        {
            temp[i] = 0;
            for (int p = crs_ptrs[i]; p < crs_ptrs[i + 1]; p++)
            {
                temp[i] += crs_vals[p] * out[crs_colids[p]];
                // temp[i] += crs_vals[p] * out[crs_colids[p]] + crs_vals[p + 1] * out[crs_colids[p + 1]] + crs_vals[p + 2] * out[crs_colids[p + 2]] + crs_vals[p + 3] * out[crs_colids[p + 3]];
                // temp[i] += crs_vals[p] * out[crs_colids[p]];
                // temp[i] += crs_vals[p + 1] * out[crs_colids[p + 1]];
                // temp[i] += crs_vals[p + 2] * out[crs_colids[p + 2]];
                // temp[i] += crs_vals[p + 3] * out[crs_colids[p + 3]];
            }
        }

// preparing for the next iteration
#pragma omp single
        {
            t = out;
            out = temp;
            temp = t;
        }
    }

    double end = omp_get_wtime();
    printf("parallel spmv ends in %f seconds\n", end - start);
}

int main()
{
    int no_iterations = 10;
    char fileName[80] = "/data/FullChip/FullChip.mtx";
    NumElements = loadDataSparse(fileName, &X, &NumRows, &NumCols);

    printf("Matrix is read, these should be equal to #rows, #columns, and #elements: %d %d %d\n", NumRows, NumCols, NumElements);

    int *crs_ptrs = (int *)malloc((NumRows + 1) * sizeof(int));
    int *crs_colids = (int *)malloc(NumElements * sizeof(int));
    double *crs_values = (double *)malloc(NumElements * sizeof(double));

    // fill the crs_ptrs array
    memset(crs_ptrs, 0, (NumRows + 1) * sizeof(int));
    for (int i = 0; i < NumElements; i++)
    {
        int rowid = X[i].i;
        if (rowid < 0 || rowid >= NumRows)
        {
            printf("problem in X, quitting - %d\n", rowid);
            exit(1);
        }
        crs_ptrs[rowid + 1]++;
    }

    // now we have cumulative ordering of crs_ptrs.
    for (int i = 1; i <= NumRows; i++)
    {
        crs_ptrs[i] += crs_ptrs[i - 1];
    }
    printf("This number should be equal to the number of nonzeros %d\n", crs_ptrs[NumRows]);

    // we set crs_colids such that for each element, it holds the related column of that element
    for (int i = 0; i < NumElements; i++)
    {
        int rowid = X[i].i;
        int index = crs_ptrs[rowid];

        crs_colids[index] = X[i].j;
        crs_values[index] = X[i].val;

        crs_ptrs[rowid] = crs_ptrs[rowid] + 1;
    }

    for (int i = NumRows; i > 0; i--)
    {
        crs_ptrs[i] = crs_ptrs[i - 1];
    }
    crs_ptrs[0] = 0;
    printf("This number should be equal to the number of nonzeros %d\n", crs_ptrs[NumRows]);

    double *in = (double *)malloc(NumRows * sizeof(double));
    for (int i = 0; i < NumRows; i++)
    {
        in[i] = 1.0f;
    }
    double *out = (double *)malloc(NumRows * sizeof(double));
    double *temp = (double *)malloc(NumRows * sizeof(double));

    iterative_spmv_seq(crs_ptrs, crs_colids, crs_values, in, no_iterations, out, temp);

    double sum = 0;
    for (int i = 0; i < NumRows; i++)
    {
        sum += out[i];
    }
    printf("the sum of the output vector for sequential SpMV is %f\n", sum);

    // Change CRS so that in each row multiples of 4 elements are stored
    // This is done to make SIMD operations easier and faster.
    // This is done by making sure crs_ptrs[i + 1] - crs_ptrs[i] is either 0 or a multiple of 4
    // When crs_ptrs[i] is increased by 1 we need to insert a 0 next to crs_vals[crs_ptrs[i] + 2] so in matrix vector multiplication it does not change the result
    // When crs_ptrs[i] is increased by 2 we need to insert a 0 next to crs_vals[crs_ptrs[i] + 1] so in matrix vector multiplication it does not change the result
    // And so on. We need to also need to insert an element to crs_colids array to make sure it corresponds to the the column of the newly inserted zero.
    // The specific column index does not matter as it is fake and will be multiplied by 0 in the matrix vector multiplication.

    // int guarentee = 4;
    // int newNumElements = 0;
    // for (int i = 0; i < NumRows; i++)
    // {
    //   int numElements = crs_ptrs[i + 1] - crs_ptrs[i];
    //   int remainder = numElements % guarentee;
    //   int numZeros = guarentee - remainder;
    //   newNumElements += numElements + numZeros;
    // }

    // int *newCrsPtrs = (int *)malloc((NumRows + 1) * sizeof(int));
    // int *newCrsColids = (int *)malloc(newNumElements * sizeof(int));
    // double *newCrsVals = (double *)_mm_malloc(newNumElements * sizeof(double), 32);

    // int newIndex = 0;
    // for (int i = 0; i < NumRows; i++)
    // {
    //   int numElements = crs_ptrs[i + 1] - crs_ptrs[i];
    //   int remainder = numElements % guarentee;
    //   int numZeros = guarentee - remainder;

    //   for (int j = crs_ptrs[i]; j < crs_ptrs[i + 1]; j++)
    //   {
    //     newCrsColids[newIndex] = crs_colids[j];
    //     newCrsVals[newIndex] = crs_values[j];
    //     newIndex++;
    //   }

    //   for (int k = 0; k < numZeros; k++)
    //   {
    //     newCrsColids[newIndex] = 0;
    //     newCrsVals[newIndex] = 0;
    //     newIndex++;
    //   }

    //   newCrsPtrs[i + 1] = newIndex;
    // }

    // free(crs_ptrs);
    // free(crs_colids);
    // free(crs_values);

    // crs_ptrs = newCrsPtrs;
    // crs_colids = newCrsColids;
    // crs_values = newCrsVals;

    double *out2 = (double *)malloc(NumRows * sizeof(double));
    for (int i = 1; i <= 16; i *= 2)
    {
        omp_set_num_threads(i);
        double start = omp_get_wtime();
        iterative_spmv_par(crs_ptrs, crs_colids, crs_values, in, no_iterations, out2, temp);
        double end = omp_get_wtime();
        printf("parallel spmv ends in %f seconds with %d threads\n", end - start, i);

        int j = 0;
        for (; j < NumRows; j++)
            if (out[i] != out2[i])
            {
                printf("\033[1;31mThe result is wrong - \033[0m");
                break;
            }

        if (j == NumRows)
            printf("\033[1;32mThe result is correct - \033[0m");

        double sum = 0;
        for (j = 0; j < NumRows; j++)
            sum += out2[j];

        printf("the sum of the output vector for the parallel computation is %f\n", sum);
    }
}
