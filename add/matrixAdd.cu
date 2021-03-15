#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
using namespace std;

__global__ void hello(){
	printf("hello?\n");
	return;
}

__global__ void mtxAddKernel(int* d_a, int* d_b, int m, int n, int* d_c){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d,%d\n", x, y);
	if(y < m && x <n){
		d_c[y*n+x] = d_a[y*n+x] + d_b[y*n+x];
	}
}

void mtxAdd(int * h_a, int* h_b, int m, int n, int* h_c){
	int size = m*n*sizeof(int);
	int* d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, size);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_b, size);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_c, size);
	dim3 dimGrid(int(ceil(n/32.0)), int(ceil(m/32.0)), 1);
	dim3 dimBlock(32, 32, 1);
	//printf("%d, %d, %d, %d\n",n, m, ceil(n/32.0), ceil(m/32.0));
	mtxAddKernel<<<dimGrid, dimBlock>>>(d_a, d_b, m, n, d_c);
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(){
	int m, n;
	scanf("%d%d", &m, &n);
	printf("%d,%d\n",m,n);
	int* h_a = (int*)malloc(m*n*sizeof(int));
	int* h_b = (int*)malloc(m*n*sizeof(int));
	int* h_c = (int*)malloc(m*n*sizeof(int));
	//memset(h_a, -1, sizeof(m*n*sizeof(int)));
	//memset(h_b, -1, sizeof(m*n*sizeof(int)));
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			h_a[i*n+j] = -1;
			h_b[i*n+j] = -1;
		}
	}
	mtxAdd(h_a, h_b, m, n, h_c);
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			if(h_c[i*n+j]!=-2){
			        printf("%d %d %d false\n",i, j, h_c[i*n+j]);
			}
			//printf("%d ", h_c[i*m+j]);
		}
	}
	return 0;
}
