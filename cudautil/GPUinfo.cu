#include <stdlib.h>
#include <stdio.h>

using namespace std;

int main(){
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaDeviceProp dev_prop;
	for(int i = 0; i < dev_count; ++i){
		printf("Device Info of Device%d\n", i);
		cudaGetDeviceProperties(&dev_prop, i);
		printf("\tmax thread per block:\t%d\n", dev_prop.maxThreadsPerBlock);
		printf("\tSM count:\t%d\n", dev_prop.multiProcessorCount);
		printf("\tclock rate\t%d\n", dev_prop.clockRate);
		printf("\tmax block dim:\t%d,%d,%d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
		printf("\tmax grid dim:\t%d,%d,%d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
		printf("\n");
	}
	return 0;
}
