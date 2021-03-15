#include <iostream>
#include <stdlib.h>
#include <string.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;

// this function will sum a large vector, the max length of the vector can be 256 * 2147483647 in nvidia 1060

__global__ void vectorSumStg1(int* DA, int* DB, int n){
    __shared__ int SA[256];
    int i = threadIdx.x;
    int blocki = blockIdx.x;
    if(i < n){
        SA[i] = DA[i];
    }else{
        SA[i] = 0;
    }
    __syncthreads();

    if(i < 128 && i+128<n){
        SA[i] += SA[i+128];
    }
    __syncthreads();

    if(i < 64 && i+64<n){
        SA[i] += SA[i+64];
    }
    __syncthreads();

    if(i < 32 && i+32<n){
        SA[i] += SA[i+32];
        SA[i] += SA[i+16];
        SA[i] += SA[i+8];
        SA[i] += SA[i+4];
        SA[i] += SA[i+2];
        SA[i] += SA[i+1];
    }
    __syncthreads();

    DB[blocki] = SA[0];
} 

__global__ void vectorSumWithinBlock(int* DA, int n, int* Das){
    extern __shared__ int SA[];
    int i = threadIdx.x;
    
    if(i < n){
        SA[i] = DA[i];
    }else{
        SA[i] = 0;
    }
    __syncthreads();

    if(i < 128 && i+128<n){
        SA[i] += SA[i+128];
    }
    __syncthreads();

    if(i < 64 && i+64<n){
        SA[i] += SA[i+64];
    }
    __syncthreads();

    if(i < 32 && i+32<n){
        SA[i] += SA[i+32];
        SA[i] += SA[i+16];
        SA[i] += SA[i+8];
        SA[i] += SA[i+4];
        SA[i] += SA[i+2];
        SA[i] += SA[i+1];
    }
    __syncthreads();

    *Das = SA[0];
}

int vectorSum(int* DA, int n){
    int Das = 0;
    
}

