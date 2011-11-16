#ifdef USE_TBB

#include "CCF_Context_TBB.h"

using namespace cv;
using namespace tbb;

CCF_Context_TBB::CCF_Context_TBB(const int& numSlices,const int& rows,const int& cols)
:CCF_Context(numSlices,rows, cols)
{
	// make sure we´re using a proper TBB timer
	if(m_timer){delete m_timer;m_timer=getTimer();} 
	
	m_taskSchedulerInit = new task_scheduler_init();
}

CCF_Context_TBB::~CCF_Context_TBB()
{
	delete m_taskSchedulerInit;

}

void CCF_Context_TBB::applyDetector(const Mat& img,Mat& detectMap)
{

	Mat imgHSV;	
	
	if (m_detectorType == SALIENCE_DETECTOR) 
		cvtColor(img, imgHSV, CV_BGR2HSV_FULL);
	
	else if(m_detectorType == FRESH_DETECTOR) 
		m_freshDetect->setImage(img);
	
	ScaleDetect scaleDetect = ScaleDetect(this,img,imgHSV,&detectMap);
	
	// do for all scales
	parallel_for(blocked_range<size_t>(0,m_detectorScales,m_detectorScales/8), scaleDetect);
	
	if (m_detectorType != HOG_DETECTOR)
		GaussianBlur(detectMap, detectMap, Size(3,3),0);

}

ScaleDetect::ScaleDetect(CCF_Context_TBB* ccf,const Mat& img,const Mat& imgHSV,Mat* detectMap):
m_ccfContext(ccf),m_img(img),m_imgHSV(imgHSV),m_detectMapPtr(detectMap)
{


}

void ScaleDetect::operator() (const blocked_range<size_t>& r) const
{
	Mat subImg;
	int top,bottom,height;
	
	vector<double> score;
	vector<Rect> found;
	
	for (size_t stripe=r.begin(); stripe!=r.end(); ++stripe)
	{
		
		top=(int)(m_img.rows* m_ccfContext->m_gridTop[stripe]);
		bottom=(int)(m_img.rows* m_ccfContext->m_gridBottom[stripe]);		
		
		//printf("height %d \n",bottom-top);
		
		height = std::max(120,bottom-top);
		
		Rect subRect = Rect(0,top, m_img.cols, height);
		subImg = Mat((m_ccfContext->m_detectorType==CCF_Context_TBB::SALIENCE_DETECTOR) ? 
					 m_imgHSV : m_img,subRect  );
		
		score.clear();
		
		//TODO: resolve magic number
		double scale=(double)(bottom-top)/128; //was 576
		
		Size winStride = Size(subImg.cols / 58 ,0),hogStride,hogPadding; // Size(24,16) for scoviz
		
		switch (m_ccfContext->m_detectorType) 
		{
			case CCF_Context_TBB::HOG_DETECTOR:
				
				hogStride = Size(2,40); //Size(max(m_img.cols/352,1),max(m_img.rows/14,1));
				hogPadding = Size(24,16); //Size(max(m_img.cols/29,1),max(m_img.rows/36,1));
				
				m_ccfContext->m_hogCCF->detectMultiScale(subImg, found, score, -10, hogStride, hogPadding, scale, 0);
				
				break;
				
			case CCF_Context_TBB::SALIENCE_DETECTOR:
				
				m_ccfContext->m_saliencyDetect->detect(subImg, score,winStride );
				break;
				
			case CCF_Context_TBB::FRESH_DETECTOR:
				m_ccfContext->m_freshDetect->detectForScale(subRect, score, winStride);
				break;
				
		}
		
		for (int i=0; i<(int)score.size(); i++)
		{		
			
			double currentScore=score[i];
			
			double avgWidth=(double) (m_ccfContext->m_detectorColumns)/score.size();
			
			int upperLimit=floor(avgWidth*(i+1));
			
			//if (i == int(score.size()-1))
			//	upperLimit=m_detectorColumns;
			
			for (int w=floor(avgWidth*i); w<upperLimit; w++) 
			{
				w = min(w, m_ccfContext->m_detectorColumns);
				
				// insert entry into detection map
				(*m_detectMapPtr).at<float>(stripe,w) = currentScore;
				
			}
		}
		
	}	
}
#endif //USE_TBB