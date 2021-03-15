#include <iostream>
#include <stdio.h>
#include <algorithm>
using namespace std;

__global__ void vecAddKernel(int* d_a, int* d_b, int n, int* d_c){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n){
		d_c[index] = d_a[index] + d_b[index];
	}
}

void vecAdd(int* a, int* b, int n, int* c){
	dim3 dimGrid(ceil(n/64), 1, 1);
	dim3 dimBlock(64, 1, 1);
	int size = n*sizeof(int);
	int* d_a;
	int* d_b;
	int* d_c;
	cudaMalloc((void**)&d_a, size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_b, size);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_c, size);
	vecAddKernel<<<dimGrid, dimBlock>>>(d_a, d_b, n, d_c);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(){
	int n;
	scanf("%d", &n);
	int vecSize = n * sizeof(int);
	int* h_a = (int*)malloc(vecSize);
	int* h_b = (int*)malloc(vecSize);
	int* h_c = (int*)malloc(vecSize);
	for(int i = 0; i < n; ++i){
		h_a[i] = i;
		h_b[i] = n-i;
	}
	vecAdd(h_a, h_b, n, h_c);
	for(int i = 0; i < n; ++i){
		printf("%d ", h_c[i]);
	}
	printf("\n");
	return 0;
}	
