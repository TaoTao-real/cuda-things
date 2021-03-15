
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

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

const int BLOCK_WIDTH = 32;


__global__ void MatrixMul1(int* A, int* B, int* C, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n && col < n){
        int sum = 0;
        for(int i = 0; i < n; ++i){
            sum += A[row*n+i]*B[i*n+col];
        }
        C[row*n+col] = sum;
    }
}

__global__ void MatrixMul2(float* DA, float* DB, float* DC, int n){
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.x;
    __shared__ float tiledAMatrix[BLOCK_WIDTH*BLOCK_WIDTH];
    __shared__ float tiledBMatrix[BLOCK_WIDTH*BLOCK_WIDTH];
    int t = (n+blockDim.x-1)/blockDim.x;
    float sum = 0;
    for(int i = 0; i < t; ++i){
        tiledAMatrix[threadIdx.x+threadIdx.y*blockDim.x] = tex2D(texRefA, row, i*blockDim.x + threadIdx.x);
        tiledBMatrix[threadIdx.x+threadIdx.y*blockDim.x] = tex2D(texRefB, i*blockDim.y+threadIdx.y, col);
        //tiledAMatrix[threadIdx.x+threadIdx.y*blockDim.x] = DA[row*n + i*blockDim.x + threadIdx.x];
        //tiledBMatrix[threadIdx.x+threadIdx.y*blockDim.x] = DB[(i*blockDim.y+threadIdx.y)*n+col];
        __syncthreads();
        for(int j = 0; j < blockDim.x; ++j){
            sum += tiledAMatrix[threadIdx.y*blockDim.x+j] * tiledBMatrix[blockDim.x*j+blockIdx.x];
        }
    }
    
    DC[row*n+col] = sum;
}

int main(){
    int n;
    cout << "input the matrix size:" << endl;
    cin >> n;
    int maxtrixSize = n*n;
    int maxtrixSizeByByte = maxtrixSize*sizeof(float);
    float* HA, * HB, * HC;
    HA = (float*) malloc(maxtrixSizeByByte);
    HB = (float*) malloc(maxtrixSizeByByte);
    HC = (float*) malloc(maxtrixSizeByByte);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            HA[i*n+j] = 1;
            HB[i*n+j] = 1;
        }
    }

    cout << "matrix filled with -1 created" << endl;

    float* DA, * DB, * DC;
    gpuErrchk(cudaMalloc((void**)&DA, maxtrixSizeByByte));
    gpuErrchk(cudaMalloc((void**)&DB, maxtrixSizeByByte));
    gpuErrchk(cudaMalloc((void**)&DC, maxtrixSizeByByte));

    gpuErrchk(cudaMemcpy(DA, HA, maxtrixSizeByByte, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(DB, HB, maxtrixSizeByByte, cudaMemcpyHostToDevice));
    
    cudaDeviceSynchronize();
    cout << "matrix loaded to GPU" << endl;

    dim3 gridDim(ceil(n/BLOCK_WIDTH), ceil(n/BLOCK_WIDTH));
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    MatrixMul2<<<gridDim, blockDim>>> (DA, DB, DC, n);
    gpuErrchk(cudaMemcpy(HC, DC, maxtrixSizeByByte, cudaMemcpyDeviceToHost));
    
    cudaDeviceSynchronize();
    cout << "matrix mul complte" << endl;

    cout << HC[0] << endl;
    return 0;
}