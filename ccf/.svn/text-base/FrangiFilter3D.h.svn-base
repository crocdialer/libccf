/*
 *  FrangiFilter3D.h
 *  PersonaBoy
 *
 *  Created by Fabian on 10/20/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef FRANGIFILTER3D_H
#define FRANGIFILTER3D_H

#include "opencv2/opencv.hpp"
#include "FrangiUtils.h"

/*
#ifdef CCF_DOUBLE_PRECISION
typedef double CCF_SCALAR
#elif
typedef float CCF_SCALAR
#endif
*/


class FrangiFilter3D 
{
private:
	
	// determine range of used sigmas
	float m_scaleRangeLow;
	float m_scaleRangeHigh;
	float m_scaleRatio;
	
	// frangi parameter for vesselness-characteristics
	float m_frangiAlpha;
	float m_frangiBeta;
	float m_frangiC;
	
	bool m_verbose;
	
	// volume dimensions
	int m_volumeDims[3] ;
	
	// stride to shift the frangi window after processing
	int m_windowStride;
	
	// working and output voxel-volumes
	cv::Mat m_voxelVolume ;
	cv::Mat m_filteredVolume ;
	
	//holds absolute maximum value
	double m_maxFilterResponse;
	double m_maxFilterHalf;	
	
	// index for next slice
	int m_currentSliceIndex;
	
	// true if stack is already build up
	bool m_isBuffering;
	
	//temporary working volumes
	
	// first order derivations of frangiVolume
	cv::Mat m_dx;
	cv::Mat m_dy;
	cv::Mat m_dz;
	
	// second order derivations of frangiVolume
	cv::Mat m_dxx;
	cv::Mat m_dyy;
	cv::Mat m_dzz;
	cv::Mat m_dxy;
	cv::Mat m_dxz;
	cv::Mat m_dyz;
	
	// compute gradients of srcVolume in direction x | y | z
	void gradient3(const cv::Mat& srcVolume, cv::Mat& dstVolume, char dim);
	
	void hessian3D(const cv::Mat& srcVolume, double sigma);
	
	// gaussian blur a volume
	void imgaussian(const cv::Mat& srcVolume, cv::Mat& dstVolume,double sigma);
	
	// reads m_dx,... and constructs lambda-arrays
	void eig3volume_array(cv::Mat& lambda1, cv::Mat& lambda2, cv::Mat& lambda3);
	
	// create working volumes of size @m_volumeDims
	void initArrays();
	
	// copy content to slice at index n in voxelVolume
	void insertSliceAtIndex(const cv::Mat& slice, const int& n);
	
	// does the actual filtering of the voxelVolume, performed every full stack
	void filterVolume(const cv::Mat& srcVolume, cv::Mat& dstVolume);
	
	inline cv::Mat getSliceAtIndex(const cv::Mat& vol, const int& n=-1)
	{
		int index = n < 0 ? m_currentSliceIndex : n ;
		
		// assert index is in range
		assert(n < m_volumeDims[0]);
		
		// this will probably always be true
		//assert(m_filteredVolume.isContinuous());
		
		//calculate offset for requested slice
		long offSet = vol.step[0] * index;
		
		return cv::Mat(m_volumeDims[1],m_volumeDims[2],vol.type(),vol.datastart + offSet);
		
	};
	
public:
	
	FrangiFilter3D(const int& numSlices,const int& rows,const int& cols,const int& stride=0);
	virtual ~FrangiFilter3D();
	
	// add a new confidenceMap to the voxelVolume
	void pushMap(const cv::Mat& confMap);
	
	inline cv::Mat getCurrentSlice(){return getSliceAtIndex(m_filteredVolume,m_currentSliceIndex-(m_volumeDims[0] - m_windowStride));};
	
	inline const int& getCurrentIndex(){return m_currentSliceIndex;};
	
	inline const bool& isBuffering(){return m_isBuffering;};
	
	void reset();
	
	void getWindowSizeAndStride(int& size,int& stride){size=m_volumeDims[0];stride=m_windowStride;};
	
	int getWindowSize(){return m_volumeDims[0];};
	void setWindowSize(const int& size);
	void setStride(const int& stride);
	
	void setFrangiParams(const float& a,const float& b,const float& c);
};

#endif // FRANGIFILTER3D_H