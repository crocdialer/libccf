/***************************************************************************
 *   original -> FrangiFilter3D.m by Dirk-Jan Kroon.                       *
 * c++ implementation written by Fabian Schmidt
 * schmifab@vision.ee.ethz.ch                                              *
 *                                                                         *
 *   http://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter
 *                                                                         *
 *   FrangiFilter3D uses the eigenvectors of the hessian to                *
 *   compute the likeliness of an image region to vessels, according       *
 *   to the method described by Frangi.                                    *
 *                                                                         *
 *                                                                         *
 *   Input:                                                                *
 *		    volume : The input image volume (vessel volume).               *
 *			.scaleRangeLow, .scaleRangeHigh : The range of sigmas used,    *
 *									          default [1 10];              *
 *			m_scaleRatio  : Step size between sigmas, default 2;           *
 *			m_frangiAlpha : Frangi vesselness constant, treshold on        *
 *					       Lambda2/Lambda3 determines if its a line        *
 *                         (vessel) or a plane-like structure,             *
 *                         default 0.5;                                    *
 *			m_frangiBeta  : Frangi vesselness constant, which determines   *
 *                         the deviation from a blob like structure,       *
 *                         default 0.5;                                    *
 *			m_frangiC     : Frangi vesselness constant which gives the     *
 *                         threshold between eigenvalues of noise and      *
 *                         vessel structure. A thumb rule is dividing      *
 *                         the greyvalues of the vessels by 4 till 6,      *
 *                         default 500;                                    *
 *                                                                         *
 *   Output: The vessel enhanced image (pixel is the maximum found in      *
 *             all scales).                                                *
 *                                                                         *
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

#include "FrangiFilter3D.h"

using namespace cv;

const double maxVfiltered = 1.6673e-008;

FrangiFilter3D::FrangiFilter3D(const int& numSlices,const int& rows,const int& cols,const int& stride):m_maxFilterResponse(maxVfiltered),
m_isBuffering(true)
{
	m_scaleRangeLow  = 3;	
	m_scaleRangeHigh = 3;	
	m_scaleRatio  = 2;
	
	// Frangi Params
	m_frangiAlpha = .6;
	m_frangiBeta  =  .6;
	m_frangiC     =  50;
	
	m_verbose    = false;
	
	// init dimensions, the voxelVolume and the slicePtr
	m_volumeDims[0] = numSlices;
	m_volumeDims[1] = rows ;
	m_volumeDims[2] = cols ;
	
	m_windowStride = stride==0 ? numSlices : stride ;
	
	initArrays();
	
}

FrangiFilter3D::~FrangiFilter3D()
{
	
}

void FrangiFilter3D::initArrays()
{
	m_currentSliceIndex = 0;
	m_isBuffering = true;
	
	m_voxelVolume = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_filteredVolume = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	
	m_dx = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dy = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dz = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dxx = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dyy = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dzz = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dxy = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dxz = cv::Mat(3,m_volumeDims,CV_64F,0.0);
	m_dyz = cv::Mat(3,m_volumeDims,CV_64F,0.0);
}

void FrangiFilter3D::pushMap(const cv::Mat& confMap)
{
	
	// fill up volume bottom up
	
	if(m_verbose) printf("frangiFilter: building up volume (%d / %d) \n",m_currentSliceIndex+1,m_volumeDims[0]);
	
	// insert confidence map into the volume
	insertSliceAtIndex(confMap,m_currentSliceIndex);
	
	// increase volumePtr
	m_currentSliceIndex ++;
	
	//stack is full
	if (m_currentSliceIndex == m_volumeDims[0]) 
	{
		m_isBuffering = false;
		
		if(m_verbose) printf("frangiFilter: processing volume...\n");
		
		// do the actual filtering of the volume
		filterVolume(m_voxelVolume, m_filteredVolume);
		
		//Sliding Window
		int shift = m_volumeDims[0] - m_windowStride;
		
		for(int i=0;i<shift;i++)
			insertSliceAtIndex(getSliceAtIndex(m_voxelVolume, m_volumeDims[0] - (shift-i)), i);
		
		// reset volumePtr
		m_currentSliceIndex = shift;
				
	}
	
	
	
}

void FrangiFilter3D::filterVolume(const cv::Mat& srcVolume, cv::Mat& dstVolume)
{
	assert(srcVolume.size[0] == m_volumeDims[0] &&
		   srcVolume.size[1] == m_volumeDims[1] &&
		   srcVolume.size[2] == m_volumeDims[2]);
	
	double scaleRangeLength = 1 + floor((m_scaleRangeHigh - m_scaleRangeLow) / m_scaleRatio);
	
	//Frangi filter for all sigmas
	for (int iterator=0; iterator<scaleRangeLength; iterator++) 
	{
		//show process
		double sigmas = m_scaleRangeLow + iterator * m_scaleRatio;
		
		if (m_verbose)
			printf("current frangi filter sigma: %.2f\n",sigmas);
		
		//calculate second order gradients (will be stored in m_dxx,...)
		hessian3D(srcVolume, sigmas);
		
		if (sigmas > 0) 
		{
			//correct for scaling
			double c = sigmas * sigmas;
			m_dxx *= c ;
			m_dyy *= c ;
			m_dzz *= c ;
			m_dxy *= c ;
			m_dxz *= c ;
			m_dyz *= c ;
		}
		
		//calculate eigenvalues and eigenvectors
		Mat lambda1=cv::Mat(3,m_volumeDims,srcVolume.type(),0.0);
		Mat lambda2=cv::Mat(3,m_volumeDims,srcVolume.type(),0.0);
		Mat lambda3=cv::Mat(3,m_volumeDims,srcVolume.type(),0.0);
		
		eig3volume_array(lambda1, lambda2, lambda3);
		
		Mat lambdaAbs1,lambdaAbs2,lambdaAbs3;
		
		//calculate absolute values of eigen values
		cv::absdiff(lambda1, Scalar::all(0), lambdaAbs1);
		cv::absdiff(lambda2, Scalar::all(0), lambdaAbs2);
		cv::absdiff(lambda3, Scalar::all(0), lambdaAbs3);
		
		//the vesselness features
		Mat Ra,Rb;
		
		//MATLAB notation
		//Ra=LambdaAbs2./LambdaAbs3;
		//Rb=LambdaAbs1./sqrt(LambdaAbs2.*LambdaAbs3);
		
		//Ra
		cv::divide(lambdaAbs2,lambdaAbs3,Ra);
		
		//Rb
		cv::multiply(lambdaAbs2,lambdaAbs3,Rb);
		cv::sqrt(Rb,Rb);
		cv::divide(lambdaAbs1,Rb,Rb);
		
		//second order structureness. S = sqrt(sum(L^2[i])) met i =< D
		Mat S;
		
		cv::multiply(lambdaAbs1,lambdaAbs1,lambdaAbs1);
		cv::multiply(lambdaAbs2,lambdaAbs2,lambdaAbs2);
		cv::multiply(lambdaAbs3,lambdaAbs3,lambdaAbs3);
		
		lambdaAbs1 += lambdaAbs2 +lambdaAbs3 ;
		
		cv::sqrt(lambdaAbs1,S);
		
		double threshold_a = 2 * m_frangiAlpha * m_frangiAlpha;
		double threshold_b = 2 * m_frangiBeta * m_frangiBeta;
		double threshold_c = 2 * m_frangiC * m_frangiC;
		
		//compute Vesselness function
		Mat expRa,expRb,expS,voxelData;
		
		// expRa		
		cv::multiply(Ra,Ra,expRa,-1/threshold_a);
		cv::exp(expRa,expRa);
		
		expRa *= -1 ;
		expRa += 1;
		
		// expRb		
		cv::multiply(Rb,Rb,expRb,-1/threshold_b);
		cv::exp(expRb,expRb);
		
		// expS		
		cv::multiply(S,S,expS,-1/threshold_c);
		cv::exp(expS,expS);
		expS *= -1 ;
		expS += 1 ;
		
		// expRa * expRb * expS
		cv::multiply(expRa,expRb,voxelData);
		cv::multiply(voxelData,expS,voxelData);
	
		MatIterator_<double>	voxIt=voxelData.begin<double>(),
								voxEnd=voxelData.end<double>(),
								lamb2It=lambda2.begin<double>(),
								lamb3It=lambda3.begin<double>();
		
		for(;voxIt!=voxEnd;voxIt++,lamb2It++,lamb3It++)
			if (*lamb2It > 0 || *lamb3It > 0)
				*voxIt = 0;
		
		//check for NaN values
		//cv::checkRange(voxelData,false);
		
		//add result of this scale to output
		if(iterator==0) 
		{
			
			dstVolume = voxelData.clone();
			
		}
		else 
		{
			//keep maximum filter response
			
			cv::max(dstVolume,voxelData,dstVolume);
		}
	}
	
	double minVal,maxVal;
	// workaround for wrong cv-assertion in 2.2 for Mat -> create 2D-header for data
	cv::minMaxLoc(cv::Mat(1,m_volumeDims[0]*m_volumeDims[1] * m_volumeDims[2],dstVolume.type(), dstVolume.datastart),&minVal,&maxVal);
	
	// keep maximum response
	m_maxFilterResponse = std::max(maxVal,m_maxFilterResponse);
	
	// and normalize volume with it
	dstVolume /= m_maxFilterResponse;
	//dstVolume /= maxVfiltered;
}

// calculate gradients on dimension 'dim'
void FrangiFilter3D::gradient3(const cv::Mat& srcVolume, cv::Mat& dstVolume, char dim)

{	
	int dimIndex = -1;
	
	switch(tolower(dim))
	{
		case 'x':
			dimIndex = 0;
			break;
		case 'y':
			dimIndex = 1;
			break;
		case 'z':
			dimIndex = 2;
			break;
			
		default:
			// return if index is invalid
			return;
	}
	
	
	// original range for whole volume
	Range ranges[3] = {Range(0,m_volumeDims[0]),Range(0,m_volumeDims[1]),Range(0,m_volumeDims[2])};
	
	// temporary headers for slice-matrices
	Mat srcLeftSlice,srcRightSlice,dstSlice ;
	
	
	// Take forward differences on left edges
	ranges[dimIndex]= cv::Range(0,1);			
	srcLeftSlice = srcVolume(ranges);
	dstSlice = dstVolume(ranges);
	
	ranges[dimIndex]= cv::Range(1,2);			
	srcRightSlice = srcVolume(ranges);
	
	//subtract: dstSlice = srcRightSlice - srcLeftSlice
	cv::subtract(srcRightSlice,srcLeftSlice,dstSlice);
	
	// Take centered differences on interior points
	for (int i=1; i < m_volumeDims[dimIndex] - 1 ; i++) 
	{
		ranges[dimIndex] = Range (i-1,i);
		srcLeftSlice = srcVolume(ranges);
		
		ranges[dimIndex] = Range(i,i+1);
		dstSlice = dstVolume(ranges);
		
		ranges[dimIndex] = Range(i+1,i+2);
		srcRightSlice = srcVolume(ranges);
		
		cv::subtract(srcRightSlice,srcLeftSlice,dstSlice);
		dstSlice /= 2.0 ;		
	}
	
	// Take forward differences on right edges
	
	ranges[dimIndex]= cv::Range(m_volumeDims[dimIndex]-2,m_volumeDims[dimIndex]-1);			
	srcLeftSlice = srcVolume(ranges);
	
	ranges[dimIndex]= cv::Range(m_volumeDims[dimIndex]-1,m_volumeDims[dimIndex]);			
	srcRightSlice = srcVolume(ranges);
	dstSlice = dstVolume(ranges);
	
	//subtract: dstSlice = srcRightSlice - srcLeftSlice
	cv::subtract(srcRightSlice,srcLeftSlice,dstSlice);
	
}



void FrangiFilter3D::hessian3D(const cv::Mat& srcVolume, double sigma)
{
	cv::Mat gaussedVolume = cv::Mat(3,m_volumeDims,CV_64F,0.0) ;
	
	if (sigma>0)
		imgaussian(srcVolume,gaussedVolume,sigma);
	else
		gaussedVolume = srcVolume.clone();
	
	// create first and second order differentiations
	gradient3(gaussedVolume, m_dz, 'z');
	gradient3(m_dz, m_dzz, 'z');
	
	gradient3(gaussedVolume, m_dy, 'y');
	gradient3(m_dy, m_dyy, 'y');
	gradient3(m_dy, m_dyz, 'z');
	
	gradient3(gaussedVolume, m_dx, 'x');
	gradient3(m_dx, m_dxx, 'x');
	gradient3(m_dx, m_dxy, 'y');
	gradient3(m_dx, m_dxz, 'z');
	
}

// Filters @srcVolume with an Gaussian filter and saves the output in @filteredVolume
void FrangiFilter3D::imgaussian(const cv::Mat& srcVolume, cv::Mat& dstVolume,double sigma)
{
	
	double kernel_size = 6 * sigma;
	int dimsI[3] = {m_volumeDims[2], m_volumeDims[1], m_volumeDims[0]};
	
	GaussianFiltering3D_double((double*)srcVolume.datastart, (double*)dstVolume.datastart, dimsI, sigma, kernel_size);
	
}

void FrangiFilter3D::eig3volume_array(cv::Mat& lambda1, cv::Mat& lambda2, cv::Mat& lambda3)
{
	double hessian[3][3];
	
	double eigenvalue[3] = {0,0,0};
	
	double** eigenvector = new double*[3];
	eigenvector[0]= new double[3];
	eigenvector[1]= new double[3];
	eigenvector[2]= new double[3];
	
	for(int i=0; i<3; i++)
		for(int j=0; j<3;j++)
			eigenvector[i][j] = hessian[i][j] = 0.0;
	
	for (int i = 0; i < m_volumeDims[0] ; i++)
		for (int j = 0; j < m_volumeDims[1]; j++)
			for (int k = 0; k < m_volumeDims[2]; k++) 
			{
				hessian[0][0]=m_dxx.at<double>(i,j,k); hessian[0][1]=m_dxy.at<double>(i,j,k); hessian[0][2]=m_dxz.at<double>(i,j,k);
				hessian[1][0]=m_dxy.at<double>(i,j,k); hessian[1][1]=m_dyy.at<double>(i,j,k); hessian[1][2]=m_dyz.at<double>(i,j,k);
				hessian[2][0]=m_dxz.at<double>(i,j,k); hessian[2][1]=m_dyz.at<double>(i,j,k); hessian[2][2]=m_dzz.at<double>(i,j,k);
				
				eigen_decomposition(hessian, eigenvector, eigenvalue);
				lambda1.at<double>(i,j,k) = eigenvalue[0];
				lambda2.at<double>(i,j,k) = eigenvalue[1];
				lambda3.at<double>(i,j,k) = eigenvalue[2];
			}
	
	delete [] eigenvector[0];
	delete [] eigenvector[1];
	delete [] eigenvector[2];
	delete [] eigenvector;
	
}

inline void FrangiFilter3D::insertSliceAtIndex(const cv::Mat& slice, const int& n)
{
	assert(slice.size().height == m_volumeDims[1] &&
		   slice.size().width == m_volumeDims[2]);
	
	MatConstIterator_<double>	srcIt = slice.begin<double>(),
	srcEnd = slice.end<double>();
	
	Mat dstSlice = getSliceAtIndex(m_voxelVolume, n);
	
	MatIterator_<double>	dstIt = dstSlice.begin<double>(),
	dstEnd = dstSlice.end<double>();
	
	for(; dstIt != dstEnd; dstIt ++, srcIt++)
		*dstIt = *srcIt;
	
}

void FrangiFilter3D::reset()
{
	//m_filteredVolume = 0.0;
	m_currentSliceIndex = 0;
	m_isBuffering = true;
	
}

void FrangiFilter3D::setWindowSize(const int& size)
{
	assert(size > 0);
	
	m_volumeDims[0] = size;
	initArrays();
}

void FrangiFilter3D::setStride(const int& stride)
{
	assert(0<stride && stride <= m_volumeDims[0]);
	
	m_windowStride = stride;
	reset();
}

void FrangiFilter3D::setFrangiParams(const float& a,const float& b,const float& c)
{
	m_frangiAlpha = a;
	m_frangiBeta = b;
	m_frangiC = c;
}