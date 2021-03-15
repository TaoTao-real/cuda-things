#include <stdio.h>

using namespace std;

__global__ void hello(){
	printf("hello world from GPU!\n");
	return;
}

int main(){
	hello<<<1,1>>>();
	cudaError_t err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("kernel launch failed with error %s\n", cudaGetErrorString(err));
	}
	return 0;
}
