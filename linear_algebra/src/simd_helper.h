#ifndef SIMD_HELPER_H
#define SIMD_HELPER_H

#include "classic_helper.h"
#include <arm_neon.h>

// Transpose function to use SIMD efficiently
template <typename T>
T **transpose_matrix(T **matrix, size_t n)
{
    T **transposed = alloc_matrix<T>(n);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Base template (unused)
template <typename T>
struct MatrixMatrixMultiply
{
    static T **apply(T **A, T **B, size_t n)
    {
        // Generic implementation or static_assert to force specialization
        static_assert(sizeof(T) == 0, "This function is only specialized for float and double.");
        return nullptr;
    }
};

// Specialization for float
template <>
struct MatrixMatrixMultiply<float>
{
    static float **apply(float **A, float **B, size_t n)
    {
        float **BT = transpose_matrix<float>(B, n); // Transpose B
        float **C = alloc_matrix<float>(n);

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                float32x4_t accVec = vdupq_n_f32(0); // Initialize accumulator vector
                size_t k;
                for (k = 0; k + 3 < n; k += 4)
                {
                    float32x4_t aVec = vld1q_f32(&A[i][k]);
                    float32x4_t bVec = vld1q_f32(&BT[j][k]);

                    accVec = vmlaq_f32(accVec, aVec, bVec); // Multiply and accumulate
                }

                // Add up the elements of the accumulator vector
                float sum = vaddvq_f32(accVec);

                // Cleanup loop for the remaining elements
                for (; k < n; ++k)
                {
                    sum += A[i][k] * BT[j][k];
                }

                C[i][j] = sum;
            }
        }

        free_matrix<float>(BT, n); // Free the transposed matrix
        return C;
    }
};

// Specialization for double
template <>
struct MatrixMatrixMultiply<double>
{
    static double **apply(double **A, double **B, size_t n)
    {
        double **BT = transpose_matrix<double>(B, n);
        double **C = alloc_matrix<double>(n);

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                float64x2_t accVec = vdupq_n_f64(0); // Initialize accumulator vector
                size_t k;
                for (k = 0; k + 2 <= n; k += 2)
                {
                    float64x2_t aVec = vld1q_f64(&A[i][k]);
                    float64x2_t bVec = vld1q_f64(&BT[j][k]);
                    accVec = vmlaq_f64(accVec, aVec, bVec); // Multiply and accumulate
                }

                // Add up the elements of the accumulator vector
                double sum = vaddvq_f64(accVec);

                // Cleanup loop for the remaining elements
                for (; k < n; ++k)
                {
                    sum += A[i][k] * BT[j][k];
                }
                C[i][j] = sum;
            }
        }

        free_matrix<double>(BT, n); // Free the transposed matrix
        return C;
    }
};

// Base template (unused)
template <typename T>
struct MatrixVectorMultiply
{
    static T *apply(T **A, T *x, size_t n)
    {
        // Generic implementation or static_assert to force specialization
        static_assert(sizeof(T) == 0, "This function is only specialized for float and double.");
        return nullptr;
    }
};

// Specialization for float
template <>
struct MatrixVectorMultiply<float>
{
    static float *apply(float **A, float *x, size_t n)
    {
        float *y = alloc_vector<float>(n);
        for (size_t i = 0; i < n; ++i)
        {
            float32x4_t sumVec = vdupq_n_f32(0); // Initialize sum vector to 0
            size_t j;
            for (j = 0; j + 3 < n; j += 4)
            {
                // Load 4 elements from the matrix and vector into Neon vectors
                float32x4_t matrixVec = vld1q_f32(&A[i][j]);
                float32x4_t vectorVec = vld1q_f32(&x[j]);

                // Multiply the matrix and vector elements
                float32x4_t mult = vmulq_f32(matrixVec, vectorVec);

                // Accumulate the results
                sumVec = vaddq_f32(sumVec, mult);
            }
            // Horizontal add to sum the elements of sumVec
            y[i] = vaddvq_f32(sumVec);

            // Cleanup loop for remaining elements when n is not a multiple of 4
            for (; j < n; ++j)
            {
                y[i] += A[i][j] * x[j];
            }
        }
        return y;
    }
};

// Specialization for double
template <>
struct MatrixVectorMultiply<double>
{
    static double *apply(double **A, double *x, size_t n)
    {
        double *y = alloc_vector<double>(n);
        for (size_t i = 0; i < n; ++i)
        {
            float64x2_t sumVec = vdupq_n_f64(0); // Initialize sum vector to 0
            size_t j;
            for (j = 0; j + 1 <= n; j += 2)
            {
                // Load 2 elements from the matrix and vector into Neon vectors
                float64x2_t matrixVec = vld1q_f64(&A[i][j]);
                float64x2_t vectorVec = vld1q_f64(&x[j]);

                // Multiply the matrix and vector elements
                float64x2_t mult = vmulq_f64(matrixVec, vectorVec);

                // Accumulate the results
                sumVec = vaddq_f64(sumVec, mult);
            }
            // Horizontal add to sum the elements of sumVec
            y[i] = vaddvq_f64(sumVec);

            // Cleanup loop for remaining elements when n is not a multiple of 2
            for (; j < n; ++j)
            {
                y[i] += A[i][j] * x[j];
            }
        }
        return y;
    }
};

#endif