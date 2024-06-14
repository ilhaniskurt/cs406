#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

#define WORD_SIZE 4 //this code only works for 4 letter words

// Kernel to count occurrences of a 4-character word in the text
__global__ void wordCount(const char* text, const char* word, int* count, int text_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= text_len - WORD_SIZE) {
        bool match = true;
        for (int i = 0; i < WORD_SIZE; ++i) {
            if (text[idx + i] != word[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            atomicAdd(count, 1);
        }
    }
}

int main(int argc, char** argv) {
    const char base_text[1024] = "CUDA is a parallel computing platform and application programming interface model created by Nvidia. CUDA gives developers access to the virtual instruction set and memory of the parallel computational elements in CUDA GPUs.";

    const char* h_word = argv[1]; //word to be searched
    const unsigned int multiplier = atoi(argv[2]); //the base_text is copied this number of times to increase the input size

    //copies the base text to create larger instances -----
    const int base_len = strlen(base_text);
    const int text_len = base_len * multiplier;
    char* h_text = new char[text_len];
    for(int i = 0; i < multiplier; i++) {
        memcpy(h_text + (i * base_len), base_text, base_len);
    }
    //data is ready in h_text -----------------------------
    
    int h_count = 0; //result variable on the host
    // Device variables 
    char *d_text, *d_word;
    int *d_count;

    // Allocate memory on the device
    cudaMalloc((void**)&d_text, text_len * sizeof(char));
    cudaMalloc((void**)&d_word, WORD_SIZE * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_text, h_text, text_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, WORD_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (text_len + blockSize - 1) / blockSize;
    wordCount<<<numBlocks, blockSize>>>(d_text, d_word, d_count, text_len);

    // Copy result back to host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    std::cout << "The word \"" << h_word << "\" appears " << h_count << " times in the text." << std::endl;

    // Clear the memory on the device
    cudaFree(d_text);
    cudaFree(d_word);
    cudaFree(d_count);

    delete[] h_text;

    return 0;
}
