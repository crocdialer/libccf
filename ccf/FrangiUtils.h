/*
 *  FrangiUtils.h
 *  CCF Library
 *
 *  Created by Fabian Schmidt on 11/11/10.
 *
 */

#ifndef FRANGIUTILS_H
#define FRANGIUTILS_H

#include <iostream>
#include "math.h"
#include <cstdlib>

#define clamp(a, b1, b2) min(max(a, b1), b2);

#ifdef MAX
#undef MAX
#endif
#define MAX(a, b) ((a)>(b)?(a):(b))

void imfilter1D_double(double *I, int lengthI, double *H, int lengthH, double *J) ;
void imfilter2D_double(double *I, int * sizeI, double *H, int lengthH, double *J) ;

void imfilter3D_double(double *I, int * sizeI, double *H, int lengthH, double *J) ;

void imfilter1D_float(float *I, int lengthI, float *H, int lengthH, float *J) ;

void imfilter2D_float(float *I, int * sizeI, float *H, int lengthH, float *J) ;

void imfilter3D_float(float *I, int * sizeI, float *H, int lengthH, float *J) ;
void imfilter2Dcolor_double(double *I, int * sizeI, double *H, int lengthH, double *J) ;

void imfilter2Dcolor_float(float *I, int * sizeI, float *H, int lengthH, float *J) ;
void GaussianFiltering3D_float(float *I, float *J, int *dimsI, double sigma, double kernel_size) ;
void GaussianFiltering2Dcolor_float(float *I, float *J, int *dimsI, double sigma, double kernel_size) ;

void GaussianFiltering2D_float(float *I, float *J, int *dimsI, double sigma, double kernel_size) ;

void GaussianFiltering1D_float(float *I, float *J, int lengthI, double sigma, double kernel_size) ;
void GaussianFiltering3D_double(double *I, double *J, int *dimsI, double sigma, double kernel_size) ;

void GaussianFiltering2Dcolor_double(double *I, double *J, int *dimsI, double sigma, double kernel_size) ;

void GaussianFiltering2D_double(double *I, double *J, int *dimsI, double sigma, double kernel_size) ;
void GaussianFiltering1D_double(double *I, double *J, int lengthI, double sigma, double kernel_size) ;

/* Eigen decomposition code for symmetric 3x3 matrices, copied from the public
 * domain Java Matrix library JAMA. */

//static double hypot2(double x, double y) { return sqrt(x*x+y*y); }

__inline double absd(double val){ if(val>0){ return val;} else { return -val;} };

/* Symmetric Householder reduction to tridiagonal form. */
//static void tred2(double** V, double d[], double e[]) ;

/* Symmetric tridiagonal QL algorithm. */
//static void tql2(double** V, double d[], double e[]) ;

void eigen_decomposition(double A[][3], double** V, double d[]) ;

#endif // FRANGIUTILS_H