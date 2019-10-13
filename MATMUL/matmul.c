#include <stdio.h>
#include <stdlib.h>
#define WIDTH 500

void matMul(float* M, float* N, float* P, int Width) 
{
    int i, j, k;
    for (i = 0; i < Width; ++i)
        for (j = 0; j < Width; ++j) {
            float sum = 0;
            for (k = 0; k < Width; ++k) {
                float a = M[i * Width + k];
                float b = N[k * Width + j];
                sum += a * b;
            }
            P[i * Width + j] = sum;
        }
}

int main()
{
    float *h_M, *h_N, *h_P;
    int i, n = WIDTH, size=sizeof(float)*n*n;
    h_P = (float *)malloc(size);
    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    for(i=0;i<n*n;i++){*(h_M+i)=(float)i; 
                     *(h_N+i)=(float)i;}
    matMul(h_M,h_N,h_P,n);
}
