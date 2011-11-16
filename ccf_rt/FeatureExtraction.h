/*
 *  FeatureExtraction.h
 *  PersonaTrainer
 *
 *  Created by Fabian on 1/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include "opencv2/core/core.hpp"

#ifdef WIN32
	#undef max
	#undef min
#endif

#ifdef USE_TBB
#include "tbb/tbb.h"
#endif

class FeatureExtraction 
{
private:
	
#ifdef USE_TBB
	tbb::spin_mutex m_spinMutex;
#endif
	
	// number of vertical subDivisions (height)
	int m_numDivisionsH;
	
	// number of horizontal subDivisions (width)
	int m_numDivisionsW;
	
	// number of (random) comparisons within subTile
	int m_numIntraComparisons;
	
	// number of (random) comparisons between diffrent tiles
	int m_numInterComparisons;
	
	// use hue inforamtion
	bool m_compareHue;
	
	int m_numStableFeatures;
	
	cv::Mat m_grayImage;
	
	std::vector<cv::Mat> m_hsvChannels;
	
	cv::RNG m_randomGen;
	
	// determines which positions to compare to each other.
	// each Vec6f has the following components:
	// (pos1_x, pos1_y, pos2_x, pos2_y, stableFeature, notUsed )	
	cv::Mat_<cv::Vec6f> m_compareTable;
	
	cv::Rect m_currentWindow;
	cv::Mat_<cv::Vec4i> m_pixelCompareTable;
	
	// feature vector to return (no allocation each time)
	cv::Mat m_currentFeatures;
	
	// treshold on (absolute) intensity difference
	int m_intesityDiffThresh;
	
	// treshold on (absolute) hue difference
	int m_hueDiffThresh;
	
	// checks for duplicates in current stable features
	inline bool sanityCheck(const float& x1,const float& y1,const float& x2,const float& y2)
	{
		
		cv::Vec2f a1(x1,y1),a2(x2,y2);
		
		float minD2 = 0.004; // dist > 2 %
		
		cv::MatConstIterator_<cv::Vec6f> compareIt = m_compareTable.begin(),
		compareEnd = m_compareTable.end();
		
		for(;compareIt!=compareEnd;compareIt++)
		{
			const cv::Vec6f& comp = *compareIt;
			
			if(comp[4] > 0)
			{
				cv::Vec2f b1(comp[0],comp[1]),b2(comp[2],comp[3]);
				
				// is comparison pair identical ?
				if( (length2(a1-b1)<minD2 && length2(a2-b2)<minD2) ||
				   (length2(a1-b2)<minD2 && length2(a2-b1)<minD2) )
				{
					//printf("comp too close (%.3f, %.3f | %.3f, %.3f) <--> (%.3f, %.3f | %.3f, %.3f)\n",
					//	   a1[0],a1[1],a2[0],a2[1],
					//	   b1[0],b1[1],b2[0],b2[1]);
					return false;
				}
			}
		}
		
		return true;
	}
	
	// central pixel-comparison algorithm here
	inline float comparePixels(const cv::Mat& hueWindow,const cv::Mat& satWindow,const cv::Mat& grayWindow,
							   const int& x1,const int& y1,const int& x2,const int& y2)
	{
		char ret = 0;
		// take intensity difference of the 2 pixels
		int intenseDiff = abs(grayWindow.at<unsigned char>(x1,y1) - grayWindow.at<unsigned char>(x2,y2));
		if(intenseDiff < m_intesityDiffThresh) ret |= 1;
		
		// minimum saturation for hue to make sense
		int minSat = 30;
		// hue difference
		if (m_compareHue && satWindow.at<unsigned char>(x1,y1) > minSat && satWindow.at<unsigned char>(x2,y2) > minSat) 
		{
			int hueDiff = abs(hueWindow.at<unsigned char>(x1,y1) - hueWindow.at<unsigned char>(x2,y2));
			if(hueDiff > 127) hueDiff = 256 - hueDiff ;
			
			if(hueDiff < m_hueDiffThresh) ret |= 2;
		}
		
		return  (float)ret ;
	};
	
	inline float length2(const cv::Vec2f& v){return v[0]*v[0] + v[1]*v[1];};
	
	void setWindow(const cv::Rect& r);
	
	// fills the compareTable with random values, only at positions not yet regarded as stable
	// so this is only needed for the training-cycles
	void fillCompareTable(const cv::Mat& featMask=cv::Mat());
	
	
public:
	
	FeatureExtraction(const int& numH=4,const int& numW=2,const int& numIntra=10,const int& numInter=64);
	FeatureExtraction(const std::string& compareTablePath);
	~FeatureExtraction();
	
	cv::Mat createFeatureVector(const cv::Rect& detectWindow,cv::Mat* pixTabPtr=NULL);
	
	inline int getFeatureCount()
	{
		if(m_compareTable.empty())
			return m_numDivisionsW * m_numDivisionsH * m_numIntraComparisons + m_numInterComparisons;
		
		return (m_compareTable.dataend - m_compareTable.datastart) / m_compareTable.elemSize(); 
	};
	
	// calculate actual pixelPositions according to compareTable windowSize
	// also sets m_pixelComapreTable
	cv::Mat buildPixelTable(const cv::Rect& wnd);
	
	// extracts intensity and hsv-channels from img
	void setImage(const cv::Mat& img);
	
	void setUseHueComparisons(bool b){m_compareHue = b;};
	
	// saves compareTable as binary file
	bool saveCompareTable(const std::string& savePath);
	
	// restores compareTable from binary data (Vec6f x numFeatures)
	bool loadCompareTable(const std::string& loadPath);
	
	void loadStandardCompareTable();
	
	int getNumStableFeatures(){return m_numStableFeatures;};
	
	cv::Mat drawComparisons(const cv::Mat& img=cv::Mat());
	
	void shuffleComparisons(const cv::Mat& varImportance, const float& partDrop);
	
};


#endif