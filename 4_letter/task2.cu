#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>

#define WORD_SIZE 4 // this code only works for 4 letter words
#define MAX_WORDS 65536 // maximum number of unique 4-character words

// Kernel to count occurrences of 4-character words in the text
__global__ void wordCount(const char* text, int text_len, int* word_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= text_len - WORD_SIZE) {
        int hash = 0;
        for (int i = 0; i < WORD_SIZE; ++i) {
            hash = hash * 31 + text[idx + i];
        }
        atomicAdd(&word_counts[hash % MAX_WORDS], 1);
    }
}

int main(int argc, char** argv) {
    const char base_text[1024] = "CUDA is a parallel computing platform and parallel application programming interface model created by Nvidia. CUDA gives developers access to the virtual instruction set and memory of the parallel computational elements in CUDA GPUs for parallel computing and parallel execution.";

    const unsigned int multiplier = atoi(argv[1]); // the base_text is copied this number of times to increase the input size

    // Copies the base text to create larger instances
    const int base_len = strlen(base_text);
    const int text_len = base_len * multiplier;
    char* h_text = new char[text_len];
    for (int i = 0; i < multiplier; i++) {
        memcpy(h_text + (i * base_len), base_text, base_len);
    }
    // Data is ready in h_text
    
    int* h_word_counts = new int[MAX_WORDS]();
    
    // Device variables 
    char* d_text;
    int* d_word_counts;

    // Allocate memory on the device
    cudaMalloc((void**)&d_text, text_len * sizeof(char));
    cudaMalloc((void**)&d_word_counts, MAX_WORDS * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_word_counts, 0, MAX_WORDS * sizeof(int));

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (text_len + blockSize - 1) / blockSize;
    wordCount<<<numBlocks, blockSize>>>(d_text, text_len, d_word_counts);

    // Copy result back to host
    cudaMemcpy(h_word_counts, d_word_counts, MAX_WORDS * sizeof(int), cudaMemcpyDeviceToHost);

    // Find the most frequent word
    std::unordered_map<int, std::string> hash_to_word;
    for (int i = 0; i <= text_len - WORD_SIZE; ++i) {
        int hash = 0;
        for (int j = 0; j < WORD_SIZE; ++j) {
            hash = hash * 31 + h_text[i + j];
        }
        hash_to_word[hash % MAX_WORDS] = std::string(h_text + i, WORD_SIZE);
    }

    int max_count = 0;
    int max_index = -1;
    for (int i = 0; i < MAX_WORDS; ++i) {
        if (h_word_counts[i] > max_count) {
            max_count = h_word_counts[i];
            max_index = i;
        }
    }

    std::string most_frequent_word = hash_to_word[max_index];

    // Display the result
    std::cout << "The most frequent word is \"" << most_frequent_word << "\" with a count of " << max_count << "." << std::endl;

    // Clear the memory on the device
    cudaFree(d_text);
    cudaFree(d_word_counts);

    delete[] h_text;
    delete[] h_word_counts;

    return 0;
}
