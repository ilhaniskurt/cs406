#include "classic_helper.h"
#include <benchmark/benchmark.h>

// Benchmark for Matrix-Matrix Multiplication
template <typename T>
void BM_MatrixMultiply(benchmark::State &state)
{
    size_t n = state.range(0);
    T **A = generate_matrix<T>(n);
    T **B = generate_matrix<T>(n);
    for (auto _ : state)
    {
        T **C = matrix_multiply<T>(A, B, n);
        benchmark::DoNotOptimize(C);
        free_matrix<T>(C, n);
    }
    free_matrix<T>(A, n);
    free_matrix<T>(B, n);
}

// Benchmark for Matrix-Vector Multiplication
template <typename T>
void BM_MatrixVectorMultiply(benchmark::State &state)
{
    size_t n = state.range(0);
    T **A = generate_matrix<T>(n);
    T *x = generate_vector<T>(n);
    for (auto _ : state)
    {
        T *y = matrix_vector_multiply<T>(A, x, n);
        benchmark::DoNotOptimize(y);
        free_vector<T>(y);
    }
    free_matrix<T>(A, n);
    free_vector<T>(x);
}

// Register benchmarks, 16 to 256 with doubling
BENCHMARK_TEMPLATE(BM_MatrixMultiply, float)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);
BENCHMARK_TEMPLATE(BM_MatrixVectorMultiply, float)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);

BENCHMARK_TEMPLATE(BM_MatrixMultiply, double)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);
BENCHMARK_TEMPLATE(BM_MatrixVectorMultiply, double)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);

BENCHMARK_MAIN();