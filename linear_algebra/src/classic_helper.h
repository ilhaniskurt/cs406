#ifndef CLASSIC_HELPER_H
#define CLASSIC_HELPER_H

#include <random>

// Allocate memory for a square matrix
template <typename T>
T **alloc_matrix(size_t n)
{
    T **matrix = new T *[n];
    for (size_t i = 0; i < n; ++i)
    {
        matrix[i] = new T[n];
    }
    return matrix;
}

// Free memory of a square matrix
template <typename T>
void free_matrix(T **matrix, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Allocate memory for a vector
template <typename T>
T *alloc_vector(size_t n)
{
    return new T[n];
}

// Free memory of a vector
template <typename T>
void free_vector(T *vector)
{
    delete[] vector;
}

// Generate a random square matrix
template <typename T>
T **generate_matrix(size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Assuming a fixed range for simplicity, could be parameterized or specialized
    std::uniform_real_distribution<T> dis(static_cast<T>(1), static_cast<T>(10));

    T **matrix = alloc_matrix<T>(n);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// Generate a random vector
template <typename T>
T *generate_vector(size_t n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Assuming a fixed range for simplicity, could be parameterized or specialized
    std::uniform_real_distribution<T> dis(static_cast<T>(1), static_cast<T>(10));

    T *vector = alloc_vector<T>(n);
    for (size_t i = 0; i < n; ++i)
    {
        vector[i] = dis(gen);
    }
    return vector;
}

// Matrix-matrix multiplication
template <typename T>
T **matrix_multiply(T **A, T **B, size_t n)
{
    T **C = alloc_matrix<T>(n);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            C[i][j] = static_cast<T>(0);
            for (size_t k = 0; k < n; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Matrix-vector multiplication
template <typename T>
T *matrix_vector_multiply(T **A, T *x, size_t n)
{
    T *y = alloc_vector<T>(n);
    for (size_t i = 0; i < n; ++i)
    {
        y[i] = static_cast<T>(0);
        for (size_t j = 0; j < n; ++j)
        {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
}

#endif
