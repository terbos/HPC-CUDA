#include <stdio.h>
#include <cuda.h>
#define VLEN 1000
void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
    int i;
    for(i=0;i<n;i++) h_C[i]=h_A[i]+h_B[i];
}

int main()
{
    int n = VLEN,i,ok;
    float h_A[VLEN], h_B[VLEN], h_C[VLEN];
    for(i=0;i<n;i++){h_A[i]=(float)i; 
                     h_B[i]=(float)(i+1);}
    vecAdd(h_A,h_B,h_C,n);
    ok = 1;
    for(i=0;i<n;i++){ ok &= (h_C[i]==(float)(2*i+1));}
    if(ok) printf("Everything worked!\n");
    else printf("Something went wrong!\n");
}
