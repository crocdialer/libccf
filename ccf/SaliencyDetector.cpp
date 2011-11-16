#include "SaliencyDetector.h"

using namespace std;
using namespace cv;

SaliencyDetector::SaliencyDetector():m_useHue(true)
{	
	m_scale = .25 ;
	
	m_allTimeMax = 0.1;
}

SaliencyDetector::~SaliencyDetector()
{
	
}

inline void SaliencyDetector::buildMasks(Mat& center,Mat& surround,const Mat& scaleImg, const int& detectorWidth, const Size& padding)
{
	Mat tmp;
	
	surround = Mat(scaleImg.rows,detectorWidth,CV_8UC1,1.0);
	center = Mat(scaleImg.rows,detectorWidth,CV_8UC1,1.0);
	
	tmp =  Mat(surround, 
			   Rect(padding.width,
					padding.height,
					detectorWidth-2*padding.width,
					surround.rows-2*padding.height));
	
	tmp = Scalar::all(0);
	
	center -= surround;
	
}

void SaliencyDetector::detect(const Mat& scaleImgHSV, vector<double>& score,
							  const Size& winStride)
{	
	
	Mat resizedImg,windowImg;
	
	Mat  centerHist,surroundHist;
	
	int detectorWidth = (int)floor(scaleImgHSV.rows / 2.f) ;
	Size padding = Size(scaleImgHSV.rows / 7,scaleImgHSV.rows/7);
	
	// scale down by factor m_scale 
	resize(scaleImgHSV, resizedImg, Size(0,0), m_scale, m_scale, INTER_LINEAR);
	Size winStrideRes = Size(winStride.width * m_scale, winStride.height * m_scale);
	Size paddingRes = Size(padding.width * m_scale, padding.height * m_scale);
	int detectorWidthRes = (int)floor(detectorWidth * m_scale);
	
	
	// build masks according to padding and height of scale
	Mat centerMask,surroundMask;
	
	buildMasks(centerMask,surroundMask,resizedImg,detectorWidthRes,paddingRes);
	
	// number of windows to slide through
	int numWindows = (resizedImg.cols - detectorWidthRes) / winStrideRes.width;
	
	//printf("img(%d,%d) numWindows: %d\n",resizedImg.cols,resizedImg.rows,numWindows);
	
	for (int i =0; i<numWindows; i++)
	{
		
		windowImg =  Mat(resizedImg, Rect( i*winStrideRes.width,0,detectorWidthRes,resizedImg.rows) );
		
		// let’s quantize hue to 30 levels
		// and the saturation to 32 levels
		int hbins = 24, vbins = 32;
		int histSize[] = {vbins,hbins};
		
		// hue varies from 0 to 179, see cvtColor
		// but opencv 2.2 introduced new flavour with full range
		float hranges[] = { 0, 256 };
		
		// value varies from 0 (black-gray-white) to
		// 255 (pure spectrum color)
		float vranges[] = { 0, 256 };
		
		const float* ranges[] = { vranges,hranges};
		
		// we compute the histogram from the 0-th
		int channels[] = {2,1};
		
		calcHist( &windowImg, 1, channels, surroundMask, // use surroundmask
				 surroundHist, m_useHue ? 2:1, histSize, ranges);
		
		calcHist( &windowImg, 1, channels, centerMask, // use centermask
				 centerHist, m_useHue ? 2:1, histSize, ranges);
		
		//normalize histograms
		cv::normalize(surroundHist,surroundHist,0,255,CV_MINMAX);
		cv::normalize(centerHist,centerHist,0,255,CV_MINMAX);
		
		double correlVal = (1 - compareHist(surroundHist, centerHist, CV_COMP_CORREL) ) / 2.f;
		
		{
			#ifdef USE_TBB
			tbb::spin_mutex::scoped_lock lock(m_spinMutex);
			#endif
			
			if(correlVal > m_allTimeMax) m_allTimeMax = correlVal;
			correlVal /= m_allTimeMax;
		}
		
		// compare using correlation -> [0,1] where 1 is maximal match
		score.push_back( correlVal );
		
		//double maxVal=0;
		//minMaxLoc(surroundHist, 0, &maxVal, 0, 0);
		
		//printf("hist: %d %d\n",surroundHist.rows,surroundHist.cols);
		
	}
	
	
}

void SaliencyDetector::buildIntegralHist(const cv::Mat& imgHSV)
{
	int hbins = 24;
	
	int integralDims[] = {imgHSV.rows+1,imgHSV.cols+1, hbins};
	Mat integral = Mat(3,integralDims,CV_64F,0.0);
	
	// hue varies from 0 to 179, see cvtColor
	// but opencv 2.2 introduced new flavour with full range
	//float hranges[] = { 0, 256 };
	//const float* ranges[] = { hranges };
	
	// we compute the histogram from the 0-th channel
	//int channels[] = {0};
	
	vector<Mat> imgChannels;
	cv::split(imgHSV,imgChannels);
	Mat hue = imgChannels[0];
	
	Mat rowSum = Mat(hbins,1,CV_64F);
	
	int bin;
	
	double q = (double)hbins/255.0;
	
	for (int r=0; r<imgHSV.rows; r++) 
	{
		rowSum = Scalar::all(0);
		
		for (int c=0; c<imgHSV.cols; c++) 
		{
			bin = (int) round( q* hue.at<char>(r,c));
			
			// increment bin
			rowSum.at<double>(bin,0) += 1.0;
			
			//integral.at<double>(r,c,bin) ;
			
			/*
			 if (r > 0)
			 ;//integral data[i][j] = integral data[i−1][j] + row sum;
			 else
			 ;//integral data[i][j] = row sum;
			 */
		}
	}
}