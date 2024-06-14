#pragma once

#include "types.hpp"

void formCompressed(sparseMatrix *X,
                    int *&crs_ptrs,
                    int *&crs_colids,
                    double *&crs_values,
                    int *&ccs_ptrs,
                    int *&ccs_rowids,
                    double *&ccs_values,
                    int n,
                    int nonzeros);

sparseMatrix *createSparseMatrix(double *matrix, int n, int nonZeroCount);