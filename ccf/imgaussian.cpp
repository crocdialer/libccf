/***************************************************************************

 *   http://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter    *
 *                                                                         *
 *   This function imgaussian filters the image with an Gaussian kernel.   *
 *                                                                         *
 ***************************************************************************/

//Copyright (c) 2009, Dirk-Jan Kroon
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without 
//modification, are permitted provided that the following conditions are 
//met:
//
//    * Redistributions of source code must retain the above copyright 
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright 
//      notice, this list of conditions and the following disclaimer in 
//      the documentation and/or other materials provided with the distribution
//      
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
//LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
//CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
//SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
//INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
//CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
//ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
//POSSIBILITY OF SUCH DAMAGE.

#include "FrangiUtils.h"


/*
#ifndef min
#define min(a,b)        ((a) < (b) ? (a): (b))
#endif
#ifndef max
#define max(a,b)        ((a) > (b) ? (a): (b))
#endif
#define clamp(a, b1, b2) min(max(a, b1), b2);
*/

#define clamp(a, b1, b2) min(max(a, b1), b2);

using namespace std;

void imfilter1D_double(double *I, int lengthI, double *H, int lengthH, double *J) {
    int x, i, index, offset;
    int b2, offset2;
    if(lengthI==1)  
    { 
        J[0]=I[0];
    }
    else
    {
        offset=(lengthH-1)/2;
        for(x=0; x<min(offset,lengthI); x++) {
            J[x]=0;
            b2=lengthI-1; offset2=x-offset;
            for(i=0; i<lengthH; i++) {
                index=clamp(i+offset2, 0, b2); J[x]+=I[index]*H[i];
            }
        }
       
        for(x=offset; x<(lengthI-offset); x++) {
            J[x]=0;
            b2=lengthI-1; offset2=x-offset;
            for(i=0; i<lengthH; i++) {
                index=i+offset2; J[x]+=I[index]*H[i];
            }
        }
       
         b2=lengthI-1; 
         for(x=max(lengthI-offset,offset); x<lengthI; x++) {
              J[x]=0;
              offset2=x-offset;
              for(i=0; i<lengthH; i++) {
                  index=clamp(i+offset2, 0, b2); J[x]+=I[index]*H[i];
             }
         }
       
    }
}

void imfilter2D_double(double *I, int * sizeI, double *H, int lengthH, double *J) {
    int y, x, i, y2;
    double *Irow, *Crow;
    int index=0, line=0;
    double *RCache;
    int *nCache;
    int hks, offset, offset2;
    RCache=(double *)malloc(lengthH*sizeI[0]*sizeof(double));
    for(i=0; i<lengthH*sizeI[0]; i++) { RCache[i]=0; }
    nCache=(int *)malloc(lengthH*sizeof(int));
    for(i=0; i<lengthH; i++) { nCache[i]=0; }
    hks=((lengthH-1)/2);
    for(y=0; y<min(hks,sizeI[1]); y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_double(Irow, sizeI[0], H, lengthH, Crow);
        index+=sizeI[0];
        if(y!=(sizeI[1]-1))
        {
            line++; if(line>(lengthH-1)) { line=0; }
        }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(y2=y; y2<hks; y2++) {
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
            
    for(y=hks; y<(sizeI[1]-1); y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_double(Irow, sizeI[0], H, lengthH, Crow);
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        line++; if(line>(lengthH-1)) { line=0; }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    for(y=max(sizeI[1]-1,hks); y<sizeI[1]; y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_double(Irow, sizeI[0], H, lengthH, Crow);
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    for(y=max(sizeI[1],hks); y<(sizeI[1]+hks); y++) {
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    free(RCache);
}

void imfilter3D_double(double *I, int * sizeI, double *H, int lengthH, double *J) {
    int z, j, i, z2;
    double *Islice, *Cslice;
    int index=0, line=0;
    double *SCache;
    int *nCache;
    int hks, offset, offset2;
    int nslice;
    nslice=sizeI[0]*sizeI[1];
    SCache=(double *)malloc(lengthH*nslice*sizeof(double));
	for(i=0; i<nslice; i++) { SCache[i]=0; }
    nCache=(int *)malloc(lengthH*sizeof(int));
    for(i=0; i<lengthH; i++) { nCache[i]=0; }
    hks=((lengthH-1)/2);
    for(z=0; z<min(hks,sizeI[2]); z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_double(Islice, sizeI, H, lengthH, Cslice);
        index+=nslice;
        if(z!=(sizeI[2]-1))
        {
            line++; if(line>(lengthH-1)) { line=0; }
        }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z2=z; z2<hks; z2++) {
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=hks; z<(sizeI[2]-1); z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_double(Islice, sizeI, H, lengthH, Cslice);
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        line++; if(line>(lengthH-1)) { line=0; }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=max(sizeI[2]-1,hks); z<sizeI[2]; z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_double(Islice, sizeI, H, lengthH, Cslice);
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=max(sizeI[2],hks); z<(sizeI[2]+hks); z++) {
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    free(SCache);
}

void imfilter1D_float(float *I, int lengthI, float *H, int lengthH, float *J) {
    int x, i, index, offset;
    int b2, offset2;
    if(lengthI==1)  
    { 
        J[0]=I[0];
    }
    else
    {
        offset=(lengthH-1)/2;
        for(x=0; x<min(offset,lengthI); x++) {
            J[x]=0;
            b2=lengthI-1; offset2=x-offset;
            for(i=0; i<lengthH; i++) {
                index=clamp(i+offset2, 0, b2); J[x]+=I[index]*H[i];
            }
        }
       
        for(x=offset; x<(lengthI-offset); x++) {
            J[x]=0;
            b2=lengthI-1; offset2=x-offset;
            for(i=0; i<lengthH; i++) {
                index=i+offset2; J[x]+=I[index]*H[i];
            }
        }
       
         b2=lengthI-1; 
         for(x=max(lengthI-offset,offset); x<lengthI; x++) {
              J[x]=0;
              offset2=x-offset;
              for(i=0; i<lengthH; i++) {
                  index=clamp(i+offset2, 0, b2); J[x]+=I[index]*H[i];
             }
         }
       
    }
}

void imfilter2D_float(float *I, int * sizeI, float *H, int lengthH, float *J) {
    int y, x, i, y2;
    float *Irow, *Crow;
    int index=0, line=0;
    float *RCache;
    int *nCache;
    int hks, offset, offset2;
    RCache=(float *)malloc(lengthH*sizeI[0]*sizeof(float));
    for(i=0; i<lengthH*sizeI[0]; i++) { RCache[i]=0; }
    nCache=(int *)malloc(lengthH*sizeof(int));
    for(i=0; i<lengthH; i++) { nCache[i]=0; }
    hks=((lengthH-1)/2);
    for(y=0; y<min(hks,sizeI[1]); y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_float(Irow, sizeI[0], H, lengthH, Crow);
        index+=sizeI[0];
        if(y!=(sizeI[1]-1))
        {
            line++; if(line>(lengthH-1)) { line=0; }
        }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(y2=y; y2<hks; y2++) {
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
            
    for(y=hks; y<(sizeI[1]-1); y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_float(Irow, sizeI[0], H, lengthH, Crow);
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        line++; if(line>(lengthH-1)) { line=0; }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    for(y=max(sizeI[1]-1,hks); y<sizeI[1]; y++) {
        Irow=&I[index];
        Crow=&RCache[line*sizeI[0]];
        imfilter1D_float(Irow, sizeI[0], H, lengthH, Crow);
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    for(y=max(sizeI[1],hks); y<(sizeI[1]+hks); y++) {
        offset=(y-hks)*sizeI[0]; offset2=nCache[0]*sizeI[0];
        for(x=0; x<sizeI[0]; x++) { J[offset+x]=RCache[offset2+x]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*sizeI[0];
            for(x=0; x<sizeI[0]; x++) { J[offset+x]+=RCache[offset2+x]*H[i]; }
        }
        index+=sizeI[0];
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    free(RCache);
}

void imfilter3D_float(float *I, int * sizeI, float *H, int lengthH, float *J) {
    int z, j, i, z2;
    float *Islice, *Cslice;
    int index=0, line=0;
    float *SCache;
    int *nCache;
    int hks, offset, offset2;
    int nslice;
    nslice=sizeI[0]*sizeI[1];
    SCache=(float *)malloc(lengthH*nslice*sizeof(float));
	for(i=0; i<nslice; i++) { SCache[i]=0; }
    nCache=(int *)malloc(lengthH*sizeof(int));
    for(i=0; i<lengthH; i++) { nCache[i]=0; }
    hks=((lengthH-1)/2);
    for(z=0; z<min(hks,sizeI[2]); z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_float(Islice, sizeI, H, lengthH, Cslice);
        index+=nslice;
        if(z!=(sizeI[2]-1))
        {
            line++; if(line>(lengthH-1)) { line=0; }
        }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z2=z; z2<hks; z2++) {
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=hks; z<(sizeI[2]-1); z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_float(Islice, sizeI, H, lengthH, Cslice);
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        line++; if(line>(lengthH-1)) { line=0; }
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=max(sizeI[2]-1,hks); z<sizeI[2]; z++) {
        Islice=&I[index];
        Cslice=&SCache[line*nslice];
        imfilter2D_float(Islice, sizeI, H, lengthH, Cslice);
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }
    for(z=max(sizeI[2],hks); z<(sizeI[2]+hks); z++) {
        offset=(z-hks)*nslice; offset2=nCache[0]*nslice;
        for(j=0; j<nslice; j++) { J[offset+j]=SCache[offset2+j]*H[0]; }
        for(i=1; i<lengthH; i++) {
            offset2=nCache[i]*nslice;
            for(j=0; j<nslice; j++) { J[offset+j]+=SCache[offset2+j]*H[i]; }
        }
        index+=nslice;
        for(i=0; i<(lengthH-1); i++) { nCache[i]=nCache[i+1]; } nCache[lengthH-1]=line;
    }

    free(SCache);
}

void imfilter2Dcolor_double(double *I, int * sizeI, double *H, int lengthH, double *J) {
	int i, index;
	for(i=0; i<sizeI[2]; i++)
	{
		index=i*(sizeI[0]*sizeI[1]);
		imfilter2D_double(&I[index], sizeI, H, lengthH, &J[index]);
	}
}

void imfilter2Dcolor_float(float *I, int * sizeI, float *H, int lengthH, float *J) {
	int i, index;
	for(i=0; i<sizeI[2]; i++)
	{
		index=i*(sizeI[0]*sizeI[1]);
		imfilter2D_float(&I[index], sizeI, H, lengthH, &J[index]);
	}
}

void GaussianFiltering3D_float(float *I, float *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x;
    float *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (float *)malloc(kernel_length*sizeof(float));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=(float)exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter3D_float(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering2Dcolor_float(float *I, float *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x;
    float *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (float *)malloc(kernel_length*sizeof(float));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=(float)exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter2Dcolor_float(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering2D_float(float *I, float *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x;
    float *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (float *)malloc(kernel_length*sizeof(float));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=(float)exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter2D_float(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering1D_float(float *I, float *J, int lengthI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x;
    float *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (float *)malloc(kernel_length*sizeof(float));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=(float)exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter1D_float(I, lengthI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering3D_double(double *I, double *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x, *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (double *)malloc(kernel_length*sizeof(double));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	/* Do the filtering */
	imfilter3D_double(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering2Dcolor_double(double *I, double *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x, *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (double *)malloc(kernel_length*sizeof(double));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter2Dcolor_double(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering2D_double(double *I, double *J, int *dimsI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x, *H, totalH=0;
	
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (double *)malloc(kernel_length*sizeof(double));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter2D_double(I, dimsI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}

void GaussianFiltering1D_double(double *I, double *J, int lengthI, double sigma, double kernel_size)
{
	int kernel_length,i;
    double x, *H, totalH=0;
    
	/* Construct the 1D gaussian kernel */
	if(kernel_size<1) { kernel_size=1; }
    kernel_length=(int)(2*ceil(kernel_size/2)+1);
	H = (double *)malloc(kernel_length*sizeof(double));
	x=-ceil(kernel_size/2);
	for (i=0; i<kernel_length; i++) { H[i]=exp(-((x*x)/(2*(sigma*sigma)))); totalH+=H[i]; x++; }
	for (i=0; i<kernel_length; i++) { H[i]/=totalH; }
	
	/* Do the filtering */
	imfilter1D_double(I, lengthI, H, kernel_length, J);
    /* Clear memory gaussian kernel */
	free(H);
}