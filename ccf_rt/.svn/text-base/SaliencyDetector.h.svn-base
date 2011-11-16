#ifndef SALIENCY_DETECTOR_H
#define SALIENCY_DETECTOR_H

#include "opencv2/imgproc/imgproc.hpp"

#ifdef WIN32
#define round(x) std::floor(x+0.5)
#endif

#ifdef USE_TBB
#include "tbb/tbb.h"
#endif


class SaliencyDetector 
{
	
private:
	
#ifdef USE_TBB
	tbb::spin_mutex m_spinMutex;
#endif
	
	bool m_useHue;
	
	double m_scale;
	
	double m_allTimeMax;
	
	void buildMasks(cv::Mat& center,cv::Mat& surround,
					const cv::Mat& scaleImg, const int& detectorWidth, const cv::Size& padding);
	
public:
	SaliencyDetector();
	virtual ~SaliencyDetector();
	
	void detect(const cv::Mat& integralImg, std::vector<double>& score,
				const cv::Size& winStride);
	
	void setScaleFactor(const float& s){m_scale = s;};
	
	void buildIntegralHist(const cv::Mat& imgHSV);
};

#endif//SALIENCY_DETECTOR_H