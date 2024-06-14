
# Running the OpenMP Version

## Compilation

The OpenMP-enabled executable is compiled using a Makefile. To compile the code, open your terminal and navigate to the directory containing the Makefile. Execute the following command:

```
make
```

This command will create a binary named `run` in the `build` directory.

## Execution

To run the compiled binary, use the following command:

```
./build/run <matrix_file>
```

Replace `<matrix_file>` with the path to your matrix file.

## Output

After the binary finishes running, it will return two numbers:

1. The calculated permanent of the given matrix.
2. The execution time in seconds.

# Running the CUDA Version

## Compilation

The CUDA-enabled executable can be compiled with a single command. Open your terminal and navigate to the directory containing the `final.cu` file. Execute the following command:

```
nvcc final.cu -O3 -o run_gpu
```

This command will create a binary named `run_gpu`.

## Execution

To run the compiled CUDA binary, use the following command:

```
./run_gpu <matrix_file>
```

Replace `<matrix_file>` with the path to your matrix file.

## Output

Similar to the OpenMP version, after the binary finishes running, it will return two numbers:

1. The calculated permanent of the given matrix.
2. The execution time in seconds.
