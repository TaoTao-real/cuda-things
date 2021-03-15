#include <iostream>
#include <stdlib.h>
#include <algorithm>

using namespace std;

texture<float, cudaTextureType2D, cudaReadModeElementType> texRefA;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefB;

__global__ void mmul_tex(float* DC, int n){
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.x;
    __shared__ float tiledAMatrix[BLOCK_WIDTH*BLOCK_WIDTH];
    __shared__ float tiledBMatrix[BLOCK_WIDTH*BLOCK_WIDTH];
    int t = (n+blockDim.x-1)/blockDim.x;
    float sum = 0;
    for(int i = 0; i < t; ++i){
        tiledAMatrix[threadIdx.x+threadIdx.y*blockDim.x] = DA[row*n + i*blockDim.x + threadIdx.x];
        tiledBMatrix[threadIdx.x+threadIdx.y*blockDim.x] = DB[(i*blockDim.y+threadIdx.y)*n+col];
        __syncthreads();
        for(int j = 0; j < blockDim.x; ++j){
            sum += tiledAMatrix[threadIdx.y*blockDim.x+j] * tiledBMatrix[blockDim.x*j+blockIdx.x];
        }
    }
    
    DC[row*n+col] = sum;
}

int main(){
    int n;
    cin >> n;
    int maxtrixSizeByByte = n*n*sizeof(float);
    float* HA = (float*) malloc(maxtrixSizeByByte);
    float* HB = (float*) malloc(maxtrixSizeByByte);
    float* HC = (float*) malloc(maxtrixSizeByByte);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            HA[i*n + j] = 1;
            HB[i*n + j] = 1;
        }
    }


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatFloat);
    cudaArray* cuArrayA;
    cudaMallocArray(&cuArrayA, &channelDesc, n, n);
    cudaMemcpyToArray(cuArrayA, 0, 0, HA, maxtrixSizeByByte, cudaMemcpyHostToDevice);
    cudaArray* cuArrayB;
    cudaMallocArray(&cuArrayB, &channelDesc, n, n);
    cudaMemcpyToArray(cuArrayB, 0, 0, HB, maxtrixSizeByByte, cudaMemcpyHostToDevice);

    cudaBindtexture(NULL, texRefA, cuArrayA, maxtrixSizeByByte);
    cudaBindtexture(NULL, texRefB, cuArrayB, maxtrixSizeByByte);


    float* DC;
    cudaMalloc(&DC, maxtrixSizeByByte);
    
    
}