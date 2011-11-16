/*
 *  CCF_Context.cpp
 *  PersonaBoy
 *
 *  Created by Fabian on 11/5/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "CCF_Context.h"

// static colorMap to output confidence-maps
Colormap CCF_Context::ms_colorMap(Colormap::JET) ;

//32 scales  
// following geometric/camera calib info is for cam3 of skoviz dataset cam3
// use calibrate grid function (TagWidget.cpp) for own definitions

static float GRID_CAM3_UPPER[]={0.0642361111111,0.0711805555556,0.0798611111111,0.0885416666667,0.0972222222222,
0.105902777778,0.114583333333,0.123263888889,0.131944444444,0.140625,0.149305555556,0.15625,0.166666666667,
0.173611111111,0.184027777778,0.190972222222,0.201388888889,0.208333333333,0.21875,0.227430555556,0.236111111111,
0.244791666667,0.251736111111,0.262152777778,0.269097222222,0.279513888889,0.286458333333,0.296875,0.303819444444,
0.314236111111,0.321180555556,0.331597222222};

static float GRID_CAM3_LOWER[]={0.321180555556,0.342013888889,0.362847222222,0.381944444444,0.402777777778,0.421875,
0.442708333333,0.461805555556,0.482638888889,0.501736111111,0.522569444444,0.543402777778,0.5625,0.583333333333,
0.602430555556,0.623263888889,0.642361111111,0.663194444444,0.682291666667,0.703125,0.722222222222,0.743055555556,
0.763888888889,0.782986111111,0.803819444444,0.822916666667,0.84375,0.862847222222,0.883680555556,0.902777777778,
0.923611111111,	0.942708333333};

// interval for values retrieved by OpenCV´s HOG-Detector
double CCF_Context::ms_hogMIN = -6.751444;
double CCF_Context::ms_hogMAX = 1.823120;

using namespace std;
using namespace cv;

CCF_Context::CCF_Context(const int& numHist,const int& rows,const int& cols):m_frameCounter(0),
m_detectorScales(rows),m_detectorColumns(cols),
m_dirtyContext(false),m_drawGrid(false),m_drawBoundingBoxes(true),
m_drawOverlayMap(false),m_boundingBoxThresh(.30),m_timer(NULL)
{
	printf("CCF -> OpenCV-Version: %s\n\n",CV_VERSION);
	
	m_hogCCF = new HOGDescriptorCCF();
	m_hogCCF->setSVMDetector(HOGDescriptorCCF::getDefaultPeopleDetector());
	
	m_saliencyDetect = new SaliencyDetector();
	
	m_freshDetect = new FreshDetector("./compareTable","./RT_PersonClassifier.yaml");
	
	m_detectorType = FRESH_DETECTOR ;
	
	m_mixtureModel = new GaussianMixture(Size(cols,rows));
	
	m_trajector = new Trajector(Size(cols,rows),numHist);
	
	m_detectionMap = Mat_<float>(rows,cols,0.0);
	m_confidenceMap = Mat_<float>(rows,cols,0.0);
	
	m_meanMap = Mat_<float>(rows,cols,0.0);
	
	m_gridTop = new float[rows] ;
	m_gridBottom = new float[rows] ;
	
	m_gridTop_tmp = NULL;
	m_gridBottom_tmp = NULL;
	
	setGrid(GRID_CAM3_UPPER, GRID_CAM3_LOWER);

	m_timer = getTimer();
}


CCF_Context::~CCF_Context()
{
	delete m_hogCCF;
	delete m_saliencyDetect;
	delete m_freshDetect;
	
	delete m_mixtureModel;
	delete m_trajector;
	
	delete[] m_gridTop;
	delete[] m_gridBottom;
	
	delete m_timer;
}

void CCF_Context::addSingleFrame(Mat& img)
{	
	// copy gridDefinition if something changed
	if(m_gridBottom_tmp && m_gridBottom_tmp)
	{
		memcpy(m_gridTop, m_gridTop_tmp, m_detectorScales*sizeof(float));
		memcpy(m_gridBottom, m_gridBottom_tmp, m_detectorScales*sizeof(float));
		delete[] m_gridBottom_tmp ;
		delete[] m_gridTop_tmp ;
		m_gridBottom_tmp = m_gridTop_tmp = NULL;
		
		m_dirtyContext = true;
		
	}
	
	// check if detector was changed
	// detector change won´t occur during processing
	if(m_detectorType_tmp != m_detectorType)
	{
		m_detectorType = m_detectorType_tmp;
		m_dirtyContext = true;
	}
	
	if(m_dirtyContext) performReset();
	
	m_timer->reset();
	
	try 
	{
		// run Detector on inputImage to generate detectionMap
		applyDetector(img,m_detectionMap);
		
	}
	catch (const exception& e ) 
	{
		try 
		{
			// workaround here to match hardcoded detector-grid
			cv::Mat resized;
			cv::resize (img, resized,Size(704,576));
			applyDetector(resized,m_detectionMap);
		}
		catch (...) 
		{
			//printf("caught (...) during CCF_Context::applyDetector() \n");
			if(m_drawGrid) drawGrid(img);
			return;
		}
		
	}
	
	m_infoStruct.m_detectorTime = m_timer->getElapsedTime();
	m_timer->reset();
	
	minMaxLoc(m_detectionMap, &m_infoStruct.m_detectorMinVal, &m_infoStruct.m_detectorMaxVal);
	if(m_detectorType==HOG_DETECTOR)
	{
		if(m_infoStruct.m_detectorMinVal < ms_hogMIN) ms_hogMIN = m_infoStruct.m_detectorMinVal;
		if(m_infoStruct.m_detectorMinVal > ms_hogMAX) ms_hogMAX = m_infoStruct.m_detectorMaxVal;
		
		//normalize interval to [0,1]
		
		min(m_detectionMap,0.0,m_detectionMap);
		m_detectionMap-=ms_hogMIN;
		//m_detectionMap/=(-ms_hogMIN);
		
		m_detectionMap/=(ms_hogMAX-ms_hogMIN);
		minMaxLoc(m_detectionMap, &m_infoStruct.m_detectorMinVal, &m_infoStruct.m_detectorMaxVal);
	}
	
	// filters the detectionMap with a gaussian mixture model to remove background detections
	m_mixtureModel->filterAlt(m_detectionMap, m_confidenceMap);
	m_infoStruct.m_gaussianTime = m_timer->getElapsedTime();
	m_timer->reset();
	
	
	//m_trajectorMap = m_confidenceMap.clone();
	m_trajector->filter(m_confidenceMap, m_trajectorMap);
	m_infoStruct.m_trajectTime = m_timer->getElapsedTime();
	
	// min/max values
	minMaxLoc(m_detectionMap,&m_infoStruct.m_detectorMinVal,&m_infoStruct.m_detectorMaxVal);
	minMaxLoc(m_confidenceMap,&m_infoStruct.m_gaussianMinVal,&m_infoStruct.m_gaussianMaxVal);
	
	//printf("( %.2f - %.2f ) \n",m_infoStruct.m_detectorMinVal,m_infoStruct.m_detectorMaxVal);
	
	// construct frameBundle
	CCF_ResultBundle bundle;
	bundle.m_detectionMap = m_detectionMap.clone();
	bundle.m_confidenceMap = m_confidenceMap.clone();
	bundle.m_trajectorMap = m_trajectorMap.clone();
	img = bundle.m_image = img.clone();
	bundle.m_frameInfo = m_infoStruct;
	
	bundle.m_detectRectangles = getDectectRectangles(m_trajectorMap,img.size());
	
	m_meanMap += m_trajectorMap;
	
	//only needed before buffering is complete 
	//(no more buffering, ciao frangi)
	m_currentBundle = bundle;
	
	// augment image with stuff
	if(m_drawGrid) drawGrid(img);
	
	if(m_drawBoundingBoxes) drawBoundingBoxes(img);
	
	if(m_drawOverlayMap)
	{
		Mat meanDet = getMeanDetections();
		overlayMap(img, meanDet);
		//overlayMap(img, bundle.m_detectionMap);
	}
	
	// increment counter
	m_frameCounter++;
}

void CCF_Context::applyDetector(const Mat& img,Mat& detectMap)
{	
	Mat subImg,imgHSV,tmp;
	double scale=0;
	vector<double> score;
	vector<Rect> found;
		
	if (m_detectorType == SALIENCE_DETECTOR) 
		cvtColor(img, imgHSV, CV_BGR2HSV_FULL);

	else if(m_detectorType == FRESH_DETECTOR) 
		m_freshDetect->setImage(img);
	

	int top,bottom,height;
	
	for (int stripes=0; stripes<m_detectorScales; stripes++)
	{
		
		top=(int)(img.rows* m_gridTop[stripes]);
		bottom=(int)(img.rows* m_gridBottom[stripes]);		
		
		//printf("height %d \n",bottom-top);
		
		height = std::max(128,bottom-top);
		
		Rect subRect = Rect(0,top, img.cols, height);
		subImg = Mat( (m_detectorType==SALIENCE_DETECTOR) ? imgHSV : img,subRect  );
		
		score.clear();
		
		//TODO: resolve magic number
		scale=(double)(bottom-top)/128; //was 576
		
		Size winStride = Size(subImg.cols / 58 ,0),hogStride,hogPadding; // Size(24,16) for scoviz
		
		switch (m_detectorType) 
		{
			case HOG_DETECTOR:
				
				hogStride = Size(2,40); //Size(max(m_img.cols/352,1),max(m_img.rows/14,1));
				hogPadding = Size(24,16); //Size(max(m_img.cols/29,1),max(m_img.rows/36,1));
				
				m_hogCCF->detectMultiScale(subImg, found, score, -10, hogStride, hogPadding, scale, 0);
				
				break;
				
			case SALIENCE_DETECTOR:
				
				m_saliencyDetect->detect(subImg, score, winStride );
				break;
				
			case FRESH_DETECTOR:
				m_freshDetect->detectForScale(subRect, score, winStride);
				break;
		}
		
		for (int i=0; i<(int)score.size(); i++)
		{		

			double currentScore=score[i];
			
			double avgWidth=(double)m_detectorColumns/score.size();
			
			int upperLimit=floor(avgWidth*(i+1));
			
			//if (i == int(score.size()-1))
			//	upperLimit=m_detectorColumns;
			
			for (int w=floor(avgWidth*i); w<upperLimit; w++) 
			{
				w = min(w, m_detectorColumns);
				
				// insert entry into detection map
				detectMap.at<float>(stripes,w) = currentScore;
				
			}
		}
		
	}	
	
	if (m_detectorType != HOG_DETECTOR)
		GaussianBlur(detectMap, detectMap, Size(3,3),0);
	
}

Mat CCF_Context::colorOutput(const Mat& confMap,const Size& outSize)
{
	if(confMap.empty()) return ms_colorMap.apply(Mat(outSize,CV_8UC1,0.0));
	Mat tmp = confMap.clone();
	
	double minVal,maxVal;
	minMaxLoc(tmp,&minVal,&maxVal);
	
	// generate colormap-output
	
	//cv::normalize(tmp,tmp,0,255,CV_MINMAX);
	if(maxVal>1) tmp/=maxVal;
	tmp *= 255.0;
	
	tmp.convertTo(tmp,CV_8UC1);
	
	Mat resized;
	resize (tmp, resized,outSize);
	tmp=ms_colorMap.apply(resized);
	
	return tmp;
	
}

void CCF_Context::drawGrid(Mat& img)
{
	
	for(int i=0;i<m_detectorScales;i++)
	{
		float v = (float)i  / (float)m_detectorScales;
		
		line(img, Point(0,img.rows * m_gridTop[i]), Point(img.cols,img.rows * m_gridTop[i]), ms_colorMap.valueFor(v),1);
		line(img, Point(0,img.rows * m_gridBottom[i]), Point(img.cols,img.rows * m_gridBottom[i]), ms_colorMap.valueFor(v), 1);
	}
}

vector<Rect> CCF_Context::getDectectRectangles(const Mat& confMap,const Size& imgSize,const float& thr)
{	
	float drawTreshold = (thr==-1)? m_boundingBoxThresh : thr;
	
	vector<Rect> rectangles;
	// generate rectangles
	for (int i=0; i<m_detectorScales; i++) 
	{
		int h = imgSize.height * (m_gridBottom[i] - m_gridTop[i]);
		int w = (int)round(h / 2.f);
		
		// take in account that sliding window produces rim on the right
		double frac = (double)(imgSize.width - 2*w/3.0) / (double) confMap.cols;
		
		if(m_detectorType!=SALIENCE_DETECTOR)
			frac = (double)(imgSize.width) / (double) confMap.cols;
		
		for (int j=0; j<m_detectorColumns; j++) 
		{
			double val; 
			if(confMap.type() == CV_32F) val = confMap.at<float>(i,j);
			else if(confMap.type() == CV_64F) val = confMap.at<double>(i,j);
			else return rectangles;			
			
			if ( val > drawTreshold)
			{
				
				int xPos = max((int)(j*frac-w/2.0),0);
				
				if(m_detectorType==HOG_DETECTOR)
					;//xPos = max((int)(j*frac),0);
				
				xPos = min(xPos,imgSize.width - (int)ceil(w/2.0));
				
				Rect r = Rect( xPos ,imgSize.height * m_gridTop[i],w,h);
				
				rectangles.push_back(r);
				
			}
		}
	}
	
	
	// group together bounding boxes
	groupRectangles(rectangles,2);
	
	// the detectors returns larger rectangles than the real objects.
	// so we slightly shrink the rectangles to get a nicer output.
	vector<Rect>::iterator it= rectangles.begin();
	for(;it!=rectangles.end();it++)
	{
		Rect& r = *it;
		
		r.x += cvRound(r.width*0.2);
		r.width = cvRound(r.width*0.6);
		
		r.y += cvRound(r.height*0.025);
		r.height = cvRound(r.height*0.725);
	}
	
	return rectangles;
	
}

void CCF_Context::drawBoundingBoxes(Mat& img)
{	
	vector<Rect> rectangles = getDectectRectangles(m_trajectorMap,img.size());
	
	for (unsigned int i=0; i<rectangles.size(); i++)
		rectangle(img, rectangles[i].tl(), rectangles[i].br(), ms_colorMap.valueFor(.7) , 1);
	
}

void CCF_Context::overlayMap(Mat& img,const Mat& map)
{
	Rect gridarea=Rect(Point(0,img.rows * m_gridTop[0]), Point(img.cols,img.rows * m_gridBottom[m_detectorScales-1]));
	Mat subImg = img(gridarea);
	
	Mat tmp = colorOutput(map,subImg.size());
	tmp = (tmp + subImg) / 2.0;
	tmp.copyTo(subImg);
	
}

void CCF_Context::setGrid(float* top,float* bottom)
{
	if( !(m_gridBottom_tmp && m_gridTop_tmp) )
	{
		m_gridBottom_tmp = new float[m_detectorScales];
		m_gridTop_tmp = new float[m_detectorScales];
	}
	
	
	memcpy(m_gridTop_tmp, top, m_detectorScales*sizeof(float));
	memcpy(m_gridBottom_tmp, bottom, m_detectorScales*sizeof(float));
	
}

void CCF_Context::setGrid(const Size& imgSize,const vector<Rect>& rectList)
{
	if(rectList.size() <= 1) return;
	
	if( !(m_gridBottom_tmp && m_gridTop_tmp))
	{
		m_gridBottom_tmp = new float[m_detectorScales];
		m_gridTop_tmp = new float[m_detectorScales];
	}
	
	float minAnchor_Y=1.f,maxAnchor_Y=0,minFoot_Y=1.f,maxFoot_Y=0;
	std::vector<Rect>::const_iterator listIt=rectList.begin();
	
	for(;listIt!=rectList.end();listIt++)
	{
		float anchorY = (float) (*listIt).y / (float)imgSize.height;
		float footY = (float)((*listIt).y + (*listIt).height) / (float)imgSize.height ;
		
		if(minAnchor_Y > anchorY) minAnchor_Y = anchorY; 
		if(maxAnchor_Y < anchorY) maxAnchor_Y = anchorY;
		
		if(minFoot_Y > footY) minFoot_Y = footY; 
		if(maxFoot_Y < footY) maxFoot_Y = footY;
	}
		
	//mean height and y-pos
	float height_mean=0, y_mean=0,x0,x1;
	
	
	for(listIt=rectList.begin();listIt!=rectList.end();listIt++)
	{
		height_mean += (*listIt).height;
		y_mean += (*listIt).y;
	}
	
	height_mean /= rectList.size();
	y_mean /= rectList.size();
	
	
	float sum1=0,sum2=0;
	for(listIt=rectList.begin();listIt!=rectList.end();listIt++)
	{
		sum1 += ((*listIt).height - height_mean) * ((*listIt).y - y_mean);
		
		sum2 += ((*listIt).y - y_mean) * ((*listIt).y - y_mean);
	}
	
	
	x1 = sum1 / sum2 ;
	
	x0 = height_mean - x1 * y_mean;
	
	float topEnd;
	
	float t0 = x0 / (float) imgSize.height;
	topEnd = (maxFoot_Y - maxAnchor_Y -t0) / x1 ;
	
	for (int i=0; i<m_detectorScales; i++)
	{
		m_gridTop_tmp[i] = i * (topEnd-minAnchor_Y) / (float)m_detectorScales + minAnchor_Y;
		
		m_gridBottom_tmp[i] = std::min(m_gridTop_tmp[i] +  x1 * m_gridTop_tmp[i] +t0, 1.f)  ;
		
		//assert(bottom[i]>top[i]);
		
		//printf("top: %.2f  bottom:%.2f\n",top[i],bottom[i]);
	}
	
	
}

Mat CCF_Context::getMeanDetections()
{
	Mat tmp = m_meanMap.clone();
	
	tmp /= (float) m_frameCounter;
	
	normalize(tmp, tmp, 0, 1, CV_MINMAX);
	
	return tmp;

}

void CCF_Context::performReset()
{
	m_frameCounter = 0;
	
	m_meanMap = 0.0;
	
	m_mixtureModel->resetBackground();

	m_dirtyContext = false;
}
