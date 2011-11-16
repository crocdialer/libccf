#ifndef FRESH_DETECTOR_H
#define FRESH_DETECTOR_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "FeatureExtraction.h"


class FreshDetector 
{
	
private:
	
	CvRTrees m_forest;
	
	FeatureExtraction* m_extractor;
	
	cv::Size m_imageSize;
	
public:
	FreshDetector(const std::string& compareTable="", const std::string& classifier="");
	virtual ~FreshDetector();
	
	// will slide a detector-window through imgStripe and save values to score
	void detectForScale(const cv::Rect& stripeRect, std::vector<double>& score,const cv::Size& winStride);
	
	inline void setImage(const cv::Mat& img){m_extractor->setImage(img);m_imageSize=img.size();};
	
	// load our stuff
	inline void loadClassifier(const std::string& clPath){m_forest.load(clPath.c_str());};
	inline void loadCompareTable(const std::string& compPath){assert(m_extractor->loadCompareTable(compPath));};
	
	void setUseHue(bool b){m_extractor->setUseHueComparisons(b);};
	
};

#endif//FRESH_DETECTOR_H