/*
 *  CCF_Context.h
 *  
 *  CCF_Context - Main class and public interface for ccf-library
 *  
 *  Created by Fabian Schmidt on 11/5/10.
 *  
 *
 */

#ifndef CCFCONTEXT_H
#define CCFCONTEXT_H

#include "opencv2/core/core.hpp"

#include <time.h>
#include <deque>

// include our diffrent detectors
#include "HOGDetect.h"
#include "SaliencyDetector.h"
#include "FreshDetector.h"

// and our filter chain
#include "GaussianMixture.h"
#include "FrangiFilter3D.h"

// needed for colormap output
#include "Colormap.h"

// dimension of used confidence maps
const int DETECTOR_SCALES = 32;
const int DETECTOR_COLUMNS = 297;

// number of slices for trajectory-volume
const int FRANGI_WINDOW = 10;


// Small utility to measure processing time
class TimerUtil
{
public:
	
	clock_t start_time;
	
	TimerUtil():start_time(clock()){};
	
	virtual void reset(){start_time=clock();};
	
	// return elapsed time since last reset in ms
	virtual double getElapsedTime()
	{
		return diffclock(clock(), start_time);
	};
	
	inline double diffclock(clock_t clock2,clock_t clock1)
	{
		double diffticks=clock2-clock1;
		double diffms=(diffticks*1000.0)/CLOCKS_PER_SEC;
		return diffms;
	};
};

// small struct to bundle together all ccf-values
struct CCF_Info
{
	// measured time values
	float m_detectorTime;
	float m_gaussianTime;
	float m_trajectTime;
	
	double m_detectorMinVal;
	double m_detectorMaxVal;
	
	double m_gaussianMinVal;
	double m_gaussianMaxVal;
	int m_gaussianFrames;
	
	double m_frangiMinVal;
	double m_frangiMaxVal;
	
	int m_frangiWindowSize;
	int m_frangiCurrentIndex;
	
};

// this struct bundles processing results
struct CCF_ResultBundle 
{
	CCF_Info m_frameInfo;
	
	cv::Mat m_image;
	cv::Mat m_detectionMap;
	cv::Mat m_confidenceMap;
	cv::Mat m_trajectorMap;
	
	std::vector<cv::Rect> m_detectRectangles;
	
};

// This class serves as public interface to the "Cascading Confidence Filtering" - Library (CCF)
class CCF_Context 
{
public:
	enum DetectorEnum{HOG_DETECTOR=0,SALIENCE_DETECTOR=1,FRESH_DETECTOR=2};
	
protected:
	
	// colormap to generate coloured ouput of confidence-maps
	static Colormap ms_colorMap;

	static double ms_hogMIN;
	static double ms_hogMAX;
	
	//total number of frames processed
	int m_frameCounter;
	
	// HOG - Person detector
	HOGDescriptorCCF* m_hogCCF;
	
	//Saliency Detector
	SaliencyDetector* m_saliencyDetect;
	
	//Fresh Detector
	FreshDetector* m_freshDetect;
	
	// type of used Detector
	DetectorEnum m_detectorType;
	
	DetectorEnum m_detectorType_tmp;
	
	// pointer to detector-grid definitions
	float* m_gridTop;
	float* m_gridBottom;
	
	float* m_gridTop_tmp;
	float* m_gridBottom_tmp;
	
	int m_detectorScales;
	int m_detectorColumns;
	
	// indicates that a reset was requested
	bool m_dirtyContext;
	
	// draw detector grid
	bool m_drawGrid;
	
	// draw boundingboxes
	bool m_drawBoundingBoxes;
	
	// overlay confidence map
	bool m_drawOverlayMap;
	
	// threshHold for boundingBox drawing [0...1]
	float m_boundingBoxThresh;
	
	// Gaussian mixture model for background substraction
	GaussianMixture* m_mixtureModel;
	
	// Frangi vessel filter for trajectories
	FrangiFilter3D* m_frangiFilter;
	
	// holds current detector output
	cv::Mat_<double> m_detectionMap;
	
	// current backgroundfiltered output
	cv::Mat_<double> m_confidenceMap;
	
	// accumulates detetection respones
	cv::Mat m_meanMap;
	
	// current frangi-filterd output
	cv::Mat_<double> m_frangiMap;
	
	TimerUtil* m_timer;
	
	// for displaying the detector grid
	void drawGrid(cv::Mat& img);
	
	// draw boundingboxes for high confidence values
	void drawBoundingBoxes(cv::Mat& img);
	
	// will hold all measured timeValues and other ccf stuff
	CCF_Info m_infoStruct;
	
	//current ResultBundle
	CCF_ResultBundle m_currentBundle;
	
	// queued frames
	std::deque<CCF_ResultBundle> m_ccfFrameQueue;
	
	void overlayMap(cv::Mat& img,const cv::Mat& map);
	
	virtual void applyDetector(const cv::Mat& img,cv::Mat& detectMap);
	
	void performReset();
	
public:
	
	CCF_Context(const int& numSlices=FRANGI_WINDOW,const int& rows=DETECTOR_SCALES,const int& cols=DETECTOR_COLUMNS);
	virtual ~CCF_Context();
	
	// most important function, feed CCF with new frames
	virtual void addSingleFrame(cv::Mat& img);
	
	// get related maps bundled together
	inline const CCF_ResultBundle& getCurrentResultBundle(){return m_currentBundle;};
	
	// utility function to convert a 1 channel map to a coloured output-map
	static cv::Mat colorOutput(const cv::Mat& confMap,const cv::Size& outSize=cv::Size(512,512));
	
	// utility to generate a list of boundingBoxes for given confMap
	std::vector<cv::Rect> getDectectRectangles(const cv::Mat& confMap,const cv::Size& imgSize,const float& thr=-1);
	
	inline const cv::Mat_<double>& getDetectionMap(){return m_detectionMap;};
	inline const cv::Mat_<double>& getConfidenceMap(){return m_confidenceMap;};
	
	FreshDetector* getFreshDetector(){return m_freshDetect;};
	
	FrangiFilter3D* getFrangiFilter(){return m_frangiFilter;};
	GaussianMixture* getMixtureModel(){return m_mixtureModel;};
	
	void setDrawGrid(bool b){m_drawGrid=b;};
	void setDrawBoundingBoxes(bool b){m_drawBoundingBoxes=b;};
	void setBoundingBoxThreshold(const float& th){m_boundingBoxThresh=th;};
	
	void setDrawOverlay(bool b){m_drawOverlayMap=b;};
	
	void setDetector(DetectorEnum dt){m_detectorType_tmp=dt;};
	DetectorEnum getDetectorType(){return m_detectorType;};
	
	// retrieve infostruct holding all measured ccf-values
	inline const CCF_Info& getCCF_Info(){return m_infoStruct;};
	
	void setGrid(float* top,float* bottom);
	void setGrid(const cv::Size& imgSize, const std::vector<cv::Rect>& rectList);
	
	cv::Size getGridSize(){return cv::Size(m_detectorColumns,m_detectorScales);};
	
	cv::Mat getMeanDetections();
	
	void reset(){m_dirtyContext = true;};
	
	virtual TimerUtil* getTimer()
	{
		return new TimerUtil();
	};
	
	
};

#endif // CCFCONTEXT_H