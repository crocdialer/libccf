/*
 *  GaussianMixture.h
 *  PersonaBoy
 *
 *  Created by Fabian on 11/5/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAUSSIAN_MIXTURE_H
#define GAUSSIAN_MIXTURE_H

#include "opencv2/opencv.hpp"

// needed for alternative implementation
#include "opencv2/video/background_segm.hpp"

class GaussianMixture 
{
private:
	
	//number of background components
	//int	m_bgComponents;		
	
	int m_numMixtureComponents;
	
	cv::Size m_size;
	
	
	//positive deviation threshold
	double m_devThresh;	
	
	//learning rate (between 0 and 1) (from paper 0.01)
	double m_learningRate;	
	
	//foreground threshold (0.25 or 0.75 in paper)
	//double m_fgThresh;	
	
	//initial standard deviation (for new components) var = 36 in paper
	double m_initialSD;		
	
	// p variable (used to update mean and sd)
	double m_p ;
	
	// mean needs to be initialized
	bool m_initMean;
	
	double m_maxValue;
	
	// rank of components (w/sd)
	std::vector<double> m_rank;
	
	// weights for components
	std::vector< cv::Mat_<double> > m_weights;
	
	// standard deviations
	std::vector< cv::Mat_<double> > m_pixSD;
	
	// pixel means
	std::vector< cv::Mat_<double> > m_pixMean;
	
	// difference of pixels from mean
	std::vector< cv::Mat_<double> > m_pixDiff;
	
	// initialize used maps
	void initArrays();
	
	// set mean with map, used to (re)initialize mean-arrays
	void setMean(const cv::Mat_<double>& newMean);
	
	
	// alternative implementation with opencv
	cv::BackgroundSubtractorMOG* m_backgroundModel;
	cv::Mat m_fgMask;
	
	
public:
	
	GaussianMixture(const cv::Size& s=cv::Size());
	virtual ~GaussianMixture();
	
	// update model and filter src-Map
	void filter(const cv::Mat_<double>& src,cv::Mat_<double>& dst);
	
	void filterAlt(const cv::Mat& src,cv::Mat& dst);
	
	void resetBackground();
	
	void setLearningRate(const float& r){m_learningRate=r;};
	
	const cv::Size& getSize(){return m_size;};
	
	void setNumComponents(const int& n){m_numMixtureComponents = n; resetBackground();};
	
};

#endif // GAUSSIAN_MIXTURE_H