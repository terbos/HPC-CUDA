#include <stdio.h>
#define VLEN 1000

__global__
void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
    float *d_A, *d_B, *d_C;
    int size=sizeof(float)*n;

    cudaMalloc((void **)&d_A,size);
    cudaMalloc((void **)&d_B,size);
    cudaMalloc((void **)&d_C,size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int n = VLEN,i,ok;
    float h_A[VLEN], h_B[VLEN], h_C[VLEN];
    for(i=0;i<n;i++){h_A[i]=(float)i; 
                     h_B[i]=(float)(i+1);
                     h_C[i]=0.0;}
    vecAdd(h_A,h_B,h_C,n);
    ok = 1;
    for(i=0;i<n;i++){ ok &= (h_C[i]==(float)(2*i+1));}
    if(ok) printf("Everything worked!\n");
    else printf("Something went wrong!\n");
}
