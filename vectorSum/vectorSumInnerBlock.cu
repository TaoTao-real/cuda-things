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

__global__ void vectorSumWithinBlock(int * DA, int n, int* Das){
    extern __shared__ int SA[];
    int i = threadIdx.x;
    
    if(i < n){
        SA[i] = DA[i];
    }
    __syncthreads();
    for(int offset = 1; offset < n; ){
        if(i < n && i+offset < n){
            SA[i]+=SA[i+offset];
        }
        __syncthreads();
        offset*=2;
    }
    *Das = SA[0];
}

__global__ void vectorSumWithinBlock2(int* DA, int n, int* Das){
    extern __shared__ int SA[];
    int i = threadIdx.x;
    
    if(i < n){
        SA[i] = DA[i];
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
    }
    __syncthreads();
    
    if(i < 16 && i+16<n){
        SA[i] += SA[i+16];
    }
    __syncthreads();

    if(i < 8 && i+8<n){
        SA[i] += SA[i+8];
    }
    __syncthreads();

    if(i < 4 && i+4<n){
        SA[i] += SA[i+4];
    }
    __syncthreads();

    if(i < 2 && i+2<n){
        SA[i] += SA[i+2];
    }
    __syncthreads();

    if(i < 1 && i+1<n){
        SA[i] += SA[i+1];
    }
    __syncthreads();

    *Das = SA[0];
}

// need to ensure n is larger than 32
__global__ void vectorSumWithinBlock3(int* DA, int n, int* Das){
    extern __shared__ int SA[];
    int i = threadIdx.x;
    
    if(i < n){
        SA[i] = DA[i];
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

int main(){
    int n;
    cout << "input the vector len" << endl;
    cin >> n;
    int vectorSizeInByte = sizeof(int)*n;

    int* HA;
    HA = (int*) malloc(vectorSizeInByte);
    memset(HA, -1, vectorSizeInByte);
    cout << "initialed the host array" << endl;

    int* DA;
    gpuErrchk(cudaMalloc((void**)&DA, vectorSizeInByte));
    gpuErrchk(cudaMemcpy(DA, HA, vectorSizeInByte, cudaMemcpyHostToDevice));
    cout << "coped the array to device" << endl;

    int* Das;
    gpuErrchk(cudaMalloc((void**)&Das, sizeof(int)));


    dim3 dimgrid(1);
    dim3 dimblock(256);
    int Has = -1;

    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ need to add the third arg because this kernel use dynamic shared memory
    vectorSumWithinBlock<<<dimgrid, dimblock, 256*sizeof(int)>>>(DA, n, Das);
    cout << "sum complted" << endl;
    cudaMemcpy(&Has, Das, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "coped the ans to host" << endl;
    cout << Has << endl;
    
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ need to add the third arg because this kernel use dynamic shared memory
    vectorSumWithinBlock2<<<dimgrid, dimblock, 256*sizeof(int)>>>(DA, n, Das);
    cout << "sum complted" << endl;
    cudaMemcpy(&Has, Das, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "coped the ans to host" << endl;
    cout << Has << endl;

    return 0;
}