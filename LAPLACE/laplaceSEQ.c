#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NSTEPS 500
#define NPTSX 200
#define NPTSY 200

void performUpdates(float *h_phi, float * h_oldphi, int *h_mask, int nptsx, int nptsy, int nsteps)
{
    int x, i, j, k;
    for(k=0;k<nsteps;++k){
        for(j=0;j<nptsy;j++)
            for(i=0;i<nptsx;i++){
                x = j*nptsx+i;
                if(h_mask[x]) h_phi[x] = 0.25f*(h_oldphi[x+1]+h_oldphi[x-1]+h_oldphi[x+nptsx]+h_oldphi[x-nptsx]);
            }
        for(j=0;j<nptsy;j++)
            for(i=0;i<nptsx;i++){
                x = j*nptsx+i;
                if(h_mask[x]) h_oldphi[x] = h_phi[x];
            }
    } 
}
   
int RGBval(float x){
    int R, B, G, pow8 = 256;
    if(x<=0.5){
        B = (int)((1.0-2.0*x)*255.0);
        G = (int)(2.0*x*255.0);
	R = 0; 
    }
    else{
        B = 0;
        G = (int)((2.0-2.0*x)*255.0);
        R = (int)((2.0*x-1.0)*255.0);
    }
    return (B+(G+R*pow8)*pow8);
}

int setup_grid (float  *h_phi, int nptsx, int nptsy, int  *h_mask)
{
    int i, j, nx2, ny2;

    for(j=0;j<nptsy;j++)
       for(i=0;i<nptsx;i++){
          h_phi[j*nptsx+i]  = 0.0;
          h_mask[j*nptsx+i] = 1;
       }

    for(i=0;i<nptsx;i++) h_mask[i] = 0;

    for(i=0;i<nptsx;i++) h_mask[(nptsy-1)*nptsx+i] = 0;

    for(j=0;j<nptsy;j++) h_mask[j*nptsx] = 0;

    for(j=0;j<nptsy;j++) h_mask[j*nptsx+nptsx-1] = 0;

    nx2 = nptsx/2;
    ny2 = nptsy/2;
    h_mask[ny2*nptsx+nx2] = 0;
    h_mask[ny2*nptsx+nx2-1] = 0;
    h_mask[(ny2-1)*nptsx+nx2] = 0;
    h_mask[(ny2-1)*nptsx+nx2-1] = 0;
    h_phi[ny2*nptsx+nx2]  = 1.0;
    h_phi[ny2*nptsx+nx2-1]  = 1.0;
    h_phi[(ny2-1)*nptsx+nx2]  = 1.0;
    h_phi[(ny2-1)*nptsx+nx2-1]  = 1.0;
    return 0;
}

int output_array (float *h_phi, int nptsx, int nptsy)
{
   int i, j, k=0;
   FILE *fp;

   
   fp = fopen("outSEQ.ps","w");
   fprintf(fp,"/picstr %d string def\n",nptsx);
   fprintf(fp,"50 50 translate\n");
   fprintf(fp,"%d %d scale\n",nptsx, nptsy);
   fprintf(fp,"%d %d 8 [%d 0 0 %d 0 %d] \n",nptsx, nptsy, nptsx, nptsy, -nptsx);
   fprintf(fp,"{currentfile 3 200 mul string readhexstring pop} bind false 3 colorimage\n");

   for(j=0;j<nptsy;j++){
        for(i=0;i<nptsx;i++,k++){
             fprintf(fp,"%06x",RGBval(h_phi[j*nptsx+i]));
             if((k+1)%10==0) fprintf(fp,"\n");
        }
   }
   fclose(fp);
   return 0;
}

int main (int argc, char *argv[])
{
   float *h_phi;
   float *h_oldphi;
   int *h_mask;
   int nsize1=sizeof(float)*NPTSX*NPTSY;
   int nsize2=sizeof(int)*NPTSX*NPTSY;

   h_phi = (float *)malloc(nsize1);
   h_oldphi = (float *)malloc(nsize1);
   h_mask = (int *)malloc(nsize2);
   setup_grid (h_oldphi, NPTSX, NPTSY, h_mask);
   performUpdates(h_phi,h_oldphi,h_mask,NPTSX,NPTSY,NSTEPS);
 
   output_array (h_phi, NPTSX, NPTSY);
 
   return 0;
}
