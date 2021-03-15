#include <iostream>
#include <stdlib.h>

using namespace std;

__global__ void partition_by_bit(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];          // value of integer at position i
    unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit

    values[i] = p_i;  

    __syncthreads();
    unsigned int T_before = plus_scan(values);
    unsigned int T_total  = values[size-1];
    unsigned int F_total  = size - T_total;
    __syncthreads();

    if ( p_i )
        values[T_before-1 + F_total] = x_i;
    else
        values[i - T_before] = x_i;
}

__device__ void radix_sort(unsigned int *values)
{
    int  bit;
    for( bit = 0; bit < 32; ++bit ){
        partition_by_bit(values, bit);
        __syncthreads();
    }
}

template<class T>
__device__ T plus_scan(T* x)
{
    unsigned int i = threadIdx.x; // id of thread executing this instance
    unsigned int n = blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2){
        T t;

        if ( i >= offset ) 
            t = x[i-offset];
        __syncthreads();

        if ( i >= offset ) 
            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]
        __syncthreads();
    }
    return x[i];
}


int main(){
    int n;
    cout << "input the array length" << endl;
    cin >> n;
    int* array;
    array = (int*)malloc(n*sizeof(int));
    for(int i = 0; i < n; ++i){
        array[i] = rand()%100;
    }
    plus_scan<<<1, n>>>(array);
    for(int i = 1; i < n; ++i){
        if(array[i]-array[i-1] < 0){
            cout << "array unsorted" << endl;
            break;
        }
    }
    cout << "array sorted" << endl;
    return 0;
}