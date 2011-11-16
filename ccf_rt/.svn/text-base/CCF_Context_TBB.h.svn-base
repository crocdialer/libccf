/*
 *  CCF_Context_TBB.h
 *  
 *  CCF_Context_TBB - Main class and public interface for ccf-library
 *  
 *  Derived class with parallel execution for detectors using Intel TBB
 *
 *  Created by Fabian Schmidt on 02/5/11.
 *  
 *
 */

#ifndef CCFCONTEXT_TBB_H
#define CCFCONTEXT_TBB_H

#ifdef USE_TBB

#include "CCF_Context.h"

#include "tbb/tbb.h"

#ifdef _MSC_VER
#include "tbb/tbbmalloc_proxy.h"
#endif

//forward declaration
class CCF_Context_TBB;

// Small utility to measure processing time
class TimerUtilTBB : public TimerUtil
{
public:
	
	tbb::tick_count m_startTick;
	
	TimerUtilTBB():m_startTick(tbb::tick_count::now()){};
	
	virtual void reset(){m_startTick=tbb::tick_count::now();};
	
	// return elapsed time since last reset in ms
	virtual double getElapsedTime()
	{
		return (tbb::tick_count::now()-m_startTick).seconds() * 1000.f;
	};

	
};

// worker body-class for TBB
class ScaleDetect 
{
private:
	
	CCF_Context_TBB* m_ccfContext;
	cv::Mat m_img;
	cv::Mat m_imgHSV;
	
	cv::Mat* m_detectMapPtr;
	
public:
	
	ScaleDetect(CCF_Context_TBB* ccf,const cv::Mat& img,const cv::Mat& imgHSV,cv::Mat* detectMap);
	
	void operator() (const tbb::blocked_range<size_t>& r) const;
};


// This class serves as public interface to the "Cascading Confidence Filtering" - Library (CCF)
class CCF_Context_TBB : public CCF_Context
{
	
protected:
	
	tbb::task_scheduler_init* m_taskSchedulerInit;
	
	void applyDetector(const cv::Mat& img,cv::Mat& detectMap);
	
public:
	
	friend class ScaleDetect;
	
	CCF_Context_TBB(const int& numSlices=TRAJECTOR_HISTORY,const int& rows=DETECTOR_SCALES,const int& cols=DETECTOR_COLUMNS);
	virtual ~CCF_Context_TBB();
	
	virtual TimerUtil* getTimer()
	{
		return new TimerUtilTBB();
	};
	
};

#endif //USE_TBB
#endif // CCFCONTEXT_TBB_H