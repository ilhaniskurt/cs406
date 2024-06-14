#include <iostream>
#include <cstring>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <chrono>

__global__ void SpaRyser(int *cptrs, int *rows, double *cvals, int dim, int nnz, double *x, double *p)
{
    __shared__ int sharedCptrs[64];
    __shared__ int sharedRows[500];
    __shared__ double sharedCvals[500];
    __shared__ float myX[256 * 40];
    int BlockDim = blockDim.x;
    int threadId = threadIdx.x;
    int tid = threadId + (blockIdx.x * BlockDim);
    int k = 0;
    for (; k < dim; k++)
    {
        myX[BlockDim * k + threadId] = x[k];
        sharedCptrs[k] = cptrs[k];
    }
    sharedCptrs[dim] = cptrs[dim];
    for (k = 0; k < nnz; k++)
    {
        sharedRows[k] = rows[k];
        sharedCvals[k] = cvals[k];
    }
    __syncthreads();
    unsigned long long int tn11 = (1ULL << (dim - 1)) - 1ULL;
    unsigned long long chunkSize = tn11 / (gridDim.x * BlockDim) + 1;
    unsigned long long myStart = tid * chunkSize + 1;
    tn11 = min(tn11, (tid + 1) * chunkSize);
    double myP = 0, prod = 1, s;
    int zeroNum = 0;
    unsigned long long int g = myStart, prevG = ((myStart - 1) ^ ((myStart - 1) >> 1)), gray, two_to_k;
    int i, j, ptr;

    for (i = 0; i < dim; i++)
    {
        if ((prevG >> i) & 1ULL)
        {
            for (ptr = sharedCptrs[i]; ptr < sharedCptrs[i + 1]; ptr++)
            {
                myX[sharedRows[ptr] * BlockDim + threadId] += sharedCvals[ptr];
            }
        }
    }
    for (i = 0; i < dim; i++)
    {
        if (myX[i * BlockDim + threadId])
            prod *= myX[i * BlockDim + threadId];
        else
            zeroNum++;
    }

    for (; g <= tn11; g++)
    {
        gray = g ^ (g >> 1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
        /**/
        two_to_k = 1;        // two_to_k = 2 raised to the k power (2^k)
        j = 0;
        while (two_to_k < (gray ^ prevG))
        {
            two_to_k <<= 1; // two_to_k is a bitmask to find location of 1
            j++;
        }
        s = (two_to_k & gray) ? +1.0 : -1.0;
        prevG = gray;
        for (i = sharedCptrs[j]; i < sharedCptrs[j + 1]; i++)
        {
            if (!myX[BlockDim * sharedRows[i] + threadId])
            {
                zeroNum--;
                myX[BlockDim * sharedRows[i] + threadId] += s * sharedCvals[i];
                prod *= myX[BlockDim * sharedRows[i] + threadId];
            }
            else
            {
                prod /= myX[BlockDim * sharedRows[i] + threadId];
                myX[BlockDim * sharedRows[i] + threadId] += s * sharedCvals[i];
                if (!myX[BlockDim * sharedRows[i] + threadId])
                    zeroNum++;
                else
                    prod *= myX[BlockDim * sharedRows[i] + threadId];
            }
        }
        if (!zeroNum)
            myP += ((2.0 * !(g & 1ULL) - 1.0) * prod);
    }
    p[tid] = myP;
}

__global__ void SkipPer(int *cptrs, int *rows, double *cvals, int *rptrs, int *cols, int dim, int nnz, double *x, double *p)
{
    int BlockDim = blockDim.x;
    __shared__ int sharedCptrs[40];
    __shared__ int sharedRows[400];
    __shared__ int sharedRptrs[40];
    __shared__ int sharedCols[400];
    __shared__ double sharedCvals[400];
    __shared__ float myX[256 * 40]; // 256 olacak
    int threadId = threadIdx.x;
    int tid = threadId + (blockIdx.x * BlockDim);
    int k = 0;
    for (; k < dim; k++)
    {
        myX[BlockDim * k + threadId] = x[k];
        sharedRptrs[k] = rptrs[k];
        sharedCptrs[k] = cptrs[k];
    }
    sharedCptrs[dim] = cptrs[dim];
    sharedRptrs[dim] = rptrs[dim];
    for (k = 0; k < nnz; ++k)
    {
        sharedRows[k] = rows[k];
        sharedCvals[k] = cvals[k];
        sharedCols[k] = cols[k];
    }
    __syncthreads();
    unsigned long long int tn11 = (1ULL << (dim - 1)) - 1ULL;
    unsigned long long chunkSize = tn11 / (gridDim.x * BlockDim) + 1;
    unsigned long long myStart = tid * chunkSize + 1;
    tn11 = min(tn11, (tid + 1) * chunkSize);
    double myP = 0, prod, s;
    unsigned long long int g = myStart, prevG = 0, grdiff, max, min;
    int i, j, ptr;
    bool skip;
    while (g <= tn11)
    {
        grdiff = (g ^ (g >> 1)) ^ (prevG ^ (prevG >> 1));
        prevG = g;
        for (j = 0; j < dim; ++j)
        {
            if ((grdiff >> j) & 1)
            {
                s = 2.0 * (((g ^ (g >> 1)) >> j) & 1) - 1.0;
                for (i = sharedCptrs[j]; i < sharedCptrs[j + 1]; ++i)
                {
                    myX[BlockDim * sharedRows[i] + threadId] += s * sharedCvals[i];
                }
            }
        }
        prod = 1.0;
        for (i = 0; i < dim; i++)
        {
            prod *= myX[BlockDim * i + threadId];
        }
        myP += (2.0 * !(g & 1) - 1.0) * prod;
        skip = false;
        max = g;
        for (i = 0; i < dim; i++)
        {
            if (!myX[BlockDim * i + threadId])
            {
                skip = true;
                min = UINT64_MAX;
                for (ptr = sharedRptrs[i]; ptr < sharedRptrs[i + 1]; ++ptr)
                {
                    j = sharedCols[ptr];
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
    p[tid] = myP;
}

typedef struct sparseMatrix
{
    int i;
    int j;
    double val;
} sparseMatrix;

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

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

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

    std::chrono::duration<double> elapsedTime;
    cudaSetDevice(1);

    if (is_binary)
    {
        const unsigned long n_threads = 2048 * 256;
        double tot[n_threads];
        int *d_cptrs;
        int *d_rows;
        double *d_cvals;
        double *d_x;
        double *d_p;
        int *d_rptrs;
        int *d_cols;
        // Start recording
        cudaMalloc(&d_cptrs, sizeof(int) * (n + 1));
        cudaMalloc(&d_rows, sizeof(int) * nonzeros);
        cudaMalloc(&d_rptrs, sizeof(int) * (n + 1));
        cudaMalloc(&d_cols, sizeof(int) * nonzeros);
        cudaMalloc(&d_cvals, sizeof(double) * nonzeros);
        cudaMalloc(&d_x, sizeof(double) * n);
        cudaMalloc(&d_p, sizeof(double) * n_threads);

        cudaMemcpy(d_cptrs, ccs_ptrs, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rows, ccs_rowids, sizeof(int) * nonzeros, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rptrs, crs_ptrs, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cols, crs_colids, sizeof(int) * nonzeros, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cvals, ccs_values, sizeof(double) * nonzeros, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, sizeof(double) * n, cudaMemcpyHostToDevice);
        auto start = std::chrono::high_resolution_clock::now();
        SkipPer<<<2048, 256>>>(d_cptrs, d_rows, d_cvals, d_rptrs, d_cols, n, nonzeros, d_x, d_p);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(tot, d_p, sizeof(double) * n_threads, cudaMemcpyDeviceToHost);
        for (int d = 0; d < n_threads; ++d)
        {
            p += tot[d];
            //std::cout<<tot[d]<<std::endl;
        }
        // Calculate the elapsed time
        elapsedTime = end - start;
    }
    else
    {
        const unsigned long n_threads = 2048 * 256;
        double tot[n_threads];
        int *d_cptrs;
        int *d_rows;
        double *d_cvals;
        double *d_x;
        double *d_p;

        cudaMalloc(&d_cptrs, sizeof(int) * (n + 1));
        cudaMalloc(&d_rows, sizeof(int) * nonzeros);
        cudaMalloc(&d_cvals, sizeof(double) * nonzeros);
        cudaMalloc(&d_x, sizeof(double) * n);
        cudaMalloc(&d_p, sizeof(double) * n_threads);

        cudaMemcpy(d_cptrs, ccs_ptrs, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rows, ccs_rowids, sizeof(int) * nonzeros, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cvals, ccs_values, sizeof(double) * nonzeros, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, sizeof(double) * n, cudaMemcpyHostToDevice);
        auto start = std::chrono::high_resolution_clock::now();
        SpaRyser<<<2048, 256>>>(d_cptrs, d_rows, d_cvals, n, nonzeros, d_x, d_p);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(tot, d_p, sizeof(double) * n_threads, cudaMemcpyDeviceToHost);
        for (int d = 0; d < n_threads; ++d)
        {
            p += tot[d];
        }
        elapsedTime = end - start;
    }

    std::cout << ((4 * (n & 1) - 2) * p) << " " << elapsedTime.count() << std::endl;

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