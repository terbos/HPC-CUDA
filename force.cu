#define BLOCK_SIZE 512
#include <stdio.h>
__global__ 
void force (float *virialArray, float *potentialArray, float *pval, float *vval, float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, float sigma, float rcut, float vrcut, float dvrc12, float dvrcut, int *head, int *list, int mx, int my, int mz, int natoms, int step, float sfx, float sfy, float sfz)
{
   float sigsq, rcutsq;
   float rxi, ryi, rzi, fxi, fyi, fzi;
   float rxij, ryij, rzij, rijsq;
   float rij, sr2, sr6, vij, wij, fij, fxij, fyij, fzij;
   float potential, virial;
   int i, icell, j, jcell, nabor;
   int xi, yi, zi, ix, jx, kx, xcell, ycell, zcell;
   __shared__ float vArray[BLOCK_SIZE];
   __shared__ float pArray[BLOCK_SIZE];
   int p_start = BLOCK_SIZE;
   sigsq  = sigma*sigma;
   rcutsq = rcut*rcut;

   potential = 0.0f;
   virial    = 0.0f;
   
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element < natoms)
   {
  	 rxi = rx[element];
  	 ryi = ry[element];
  	 rzi = rz[element];
  	 fxi = 0.0f;
  	 fyi = 0.0f;
  	 fzi = 0.0f;
  	 xi = (int)((rxi+0.5f)/sfx) + 1;
  	 yi = (int)((ryi+0.5f)/sfy) + 1;
  	 zi = (int)((rzi+0.5f)/sfz) + 1;
           if(xi > mx) xi = mx;
           if(yi > my) yi = my;
           if(zi > mz) zi = mz;
  	   icell = xi + (mx+2)*(yi+zi*(my+2));
           for (ix=-1;ix<=1;ix++)
               for (jx=-1;jx<=1;jx++)
                   for (kx=-1;kx<=1;kx++){
  		     xcell = ix+xi;
  		     ycell = jx+yi;
  		     zcell = kx+zi;
                       jcell = xcell + (mx+2)*(ycell+(my+2)*zcell);
                       j = head[jcell];
                       while (j>=0) {
                           if (j!=element) {
                               rxij = rxi - rx[j];
                               ryij = ryi - ry[j];
                               rzij = rzi - rz[j];
                               rijsq = rxij*rxij + ryij*ryij + rzij*rzij;
                               if (rijsq < rcutsq) {
  			                           //START FORCE_IJ
                                    
                                    rij = (float) sqrt ((float)rijsq);
                                    sr2 = sigsq/rijsq;
                                    sr6 = sr2*sr2*sr2;
                                    vij = sr6*(sr6-1.0f) - vrcut - dvrc12*(rij-rcut);
                                    wij = sr6*(sr6-0.5f) + dvrcut*rij;
                                    fij = wij/rijsq;
                                    fxij = fij*rxij;
                                    fyij = fij*ryij;
                                    fzij = fij*rzij;
                                   //END FORCE_IJ
                                   wij *= 0.5f;
                                   vij *= 0.5f;
                                   potential += vij;
                                   virial    += wij;
                                   fxi       += fxij;
                                   fyi       += fyij;
                                   fzi       += fzij;
                               }
  			 }
                           j = list[j];
                        }
  	         }
           *(fx+element) = 48.0f*fxi;
           *(fy+element) = 48.0f*fyi;
           *(fz+element) = 48.0f*fzi;

            vArray[threadIdx.x] = virial;
            pArray[threadIdx.x] = potential;
            unsigned int stride;
            unsigned int t = threadIdx.x;
            __syncthreads();
            if (t == 0)
            {
             // __syncthreads();
              for(stride = 1; stride < blockDim.x; stride += 1)
              {
                vArray[t]+= vArray[stride];
                pArray[t]+= pArray[stride];
              }
            }
            //__syncthreads();
            if(t == 0)
            {
              virialArray[blockIdx.x] = vArray[0];
              potentialArray[blockIdx.x] = pArray[0];
            }


     }

}
