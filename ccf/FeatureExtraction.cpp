/*
 *  FeatureExtraction.cpp
 *  PersonaTrainer
 *
 *  Created by Fabian on 1/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatureExtraction.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "Colormap.h"

#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;


FeatureExtraction::FeatureExtraction(const int& numH,const int& numW,const int& numIntra,const int& numInter):
m_numDivisionsH(numH),m_numDivisionsW(numW),m_numIntraComparisons(numIntra),m_numInterComparisons(numInter),
m_compareHue(true),m_numStableFeatures(0),
m_intesityDiffThresh(25),m_hueDiffThresh(10)
{
	m_compareTable = Mat_<Vec6f> (getFeatureCount(),1,Vec6f::all(0.0));
	
	m_currentFeatures = Mat (1,getFeatureCount(),CV_32F);
	
	// init RNG
	time_t rawtime;
	time ( &rawtime );
	m_randomGen = RNG(rawtime);
	
	
	// init our compareTable
	if(getFeatureCount() != 144) fillCompareTable();
	
	else loadStandardCompareTable();
	
}

FeatureExtraction::FeatureExtraction(const std::string& compareTablePath):
m_numDivisionsH(0),m_numDivisionsW(0),m_numIntraComparisons(0),m_numInterComparisons(0),m_numStableFeatures(0),
m_intesityDiffThresh(25),m_hueDiffThresh(10)
{
	m_currentFeatures = Mat (1,getFeatureCount(),CV_32F);
	
	// load compareTbale from file
	if(!loadCompareTable(compareTablePath))
		throw std::exception();
	
}

FeatureExtraction::~FeatureExtraction()
{
	
}

void FeatureExtraction::fillCompareTable(const Mat& featMask)
{
	if(featMask.size() == Size(getFeatureCount(),1))
	{
		printf("setting featureMask\n");
		
		MatConstIterator_<float> maskIt=featMask.begin<float>();
		MatIterator_<Vec6f> compareIt = m_compareTable.begin(),
		compareEnd = m_compareTable.end();
		
		for(;compareIt!=compareEnd;compareIt++,maskIt++)
		{
			if (*maskIt > 0.f) 
				(*compareIt)[4] = 1.f;
			else if (*maskIt < 0.f)
				(*compareIt)[4] = -1.f;
			
		}
	}
	
	float divH = 1.0f / (float) m_numDivisionsH ;
	float divW = 1.0f / (float) m_numDivisionsW ;
	
	Vec6f* compVec;
	
	float x1,y1,x2,y2;
	
	m_numStableFeatures = 0;
	
	// intra-Comparisons within subTiles 
	for (int r=0; r<m_numDivisionsH; r++) 
	{
		float floorY = r*divH;
		
		for (int c=0; c<m_numDivisionsW; c++)
		{
			float floorX = c*divW;
			
			int baseIndexForTile = (r*m_numDivisionsW + c)*m_numIntraComparisons;
			
			for (int i=0; i<m_numIntraComparisons; i++) 
			{
				
				compVec = m_compareTable[ baseIndexForTile + i];
				
				// comparison already stable ? -> leave it alone
				if( (*compVec)[4] <= 0 )
				{
					do 
					{
						x1 = m_randomGen.uniform(floorX,floorX+divW);
						y1 = m_randomGen.uniform(floorY,floorY+divH);
						x2 = m_randomGen.uniform(floorX,floorX+divW);
						y2 = m_randomGen.uniform(floorY,floorY+divH);
						
					} while (!sanityCheck(x1,y1,x2,y2));
					
					*compVec = Vec6f(x1,	// pos1X
									 y1,	// pos1Y
									 x2,	// pos2X
									 y2,	// pos2Y
									 0.0,	// stable ? 0 | 1 
									 (*compVec)[5]);	// variable importance
				}
				else
					m_numStableFeatures++;
			}
			
		}
	}
	
	// inter-Comparisons of (possibly) diffrent subTiles
	int baseIndex = m_numDivisionsH*m_numDivisionsW*m_numIntraComparisons;
	
	for (int i=0; i<m_numInterComparisons; i++) 
	{
		compVec = m_compareTable[ baseIndex + i];
		
		// comparison already stable ? -> leave it alone
		if( (*compVec)[4] <= 0 )
		{
			do 
			{
				x1 = m_randomGen.uniform(0.f,1.f);
				y1 = m_randomGen.uniform(0.f,1.f);
				x2 = m_randomGen.uniform(0.f,1.f);
				y2 = m_randomGen.uniform(0.f,1.f);
				
			} while (!sanityCheck(x1,y1,x2,y2));
			
			*compVec = Vec6f(x1,	// pos1X
							 y1,	// pos1Y
							 x2,	// pos2X
							 y2,	// pos2Y
							 0.0,	// stable ? 0 | 1 
							 (*compVec)[5]);	// unused
		}
		else
			m_numStableFeatures++;
	}
	
	// build according pixelTable
	//buildPixelTable();
}

Mat FeatureExtraction::buildPixelTable(const Rect& wnd)
{
	
	Mat pixTab = Mat_<Vec4i>(getFeatureCount(),1);
	
	MatIterator_<Vec4i> pixelIt = pixTab.begin<Vec4i>(),
	pixelEnd = pixTab.end<Vec4i>();
	
	MatConstIterator_<Vec6f> compareIt = m_compareTable.begin(),
	compareEnd = m_compareTable.end();
	
	for (; pixelIt != pixelEnd; pixelIt++,compareIt++) 
	{
		(*pixelIt)[0] = (int)floor( (*compareIt)[0] * wnd.width);
		(*pixelIt)[1] = (int)floor( (*compareIt)[1] * wnd.height);
		(*pixelIt)[2] = (int)floor( (*compareIt)[2] * wnd.width);
		(*pixelIt)[3] = (int)floor( (*compareIt)[3] * wnd.height);
		
	}
	
	//m_pixelCompareTable = pixTab;
	
	return pixTab;
	
	//cout << m_pixelCompareTable <<"\n";
}

void FeatureExtraction::setImage(const Mat& img)
{
	Mat tmp,hsv;	
	
	//TODO: find best value
	// blur image using BoxFilter (Gaussianblur !?) 
	blur(img, tmp, Size(3,3));
	
	//GaussianBlur(img, tmp, Size(17,17),0);
	
	// Grayscale image
	//cvtColor (tmp, m_grayImage, CV_BGR2GRAY);
	
	// HSV image
	cvtColor (tmp, hsv, CV_BGR2HSV_FULL);
	
	// split channels
	split(hsv, m_hsvChannels);
	
}

void FeatureExtraction::setWindow(const Rect& r)
{
#ifdef USE_TBB
	tbb::spin_mutex::scoped_lock lock(m_spinMutex);
#endif
	
	m_currentWindow = r;
	
}

//TODO: parallize with TBB !?
Mat FeatureExtraction::createFeatureVector(const Rect& detectWindow,Mat* pixTabPtr)
{
	Mat pixTab;
	Mat featureVec = Mat (1,getFeatureCount(),CV_32F);
	
	if(detectWindow.size() != m_currentWindow.size() && !pixTabPtr)
	{
		pixTab = buildPixelTable(detectWindow);
		pixTabPtr = &pixTab;
	}
	
	//setWindow(detectWindow);
	
	
	Mat hueWnd=m_hsvChannels[0](detectWindow),
	satWnd=m_hsvChannels[1](detectWindow),
	grayWnd=m_hsvChannels[2](detectWindow);
	
	// retrieve pixel values and do comparisons
	
	MatIterator_<float> featureIt = featureVec.begin<float>(),
	featureEnd = featureVec.end<float>();
	
	MatConstIterator_<Vec4i> pixelIt = pixTabPtr->begin<Vec4i>(),
	pixelEnd = pixTabPtr->end<Vec4i>();
	
	int x1,y1,x2,y2;
	
	for (; featureIt != featureEnd; featureIt++,pixelIt++) 
	{
		x1 = (*pixelIt)[0] ;
		y1 = (*pixelIt)[1] ;
		x2 = (*pixelIt)[2] ;
		y2 = (*pixelIt)[3] ;
		
		// write result to featurevector
		*featureIt = comparePixels(hueWnd,satWnd,grayWnd,x1,y1,x2,y2) ;
	}
	
	//cout << m_currentFeatures <<"\n";
	
	return featureVec;
	
}

bool FeatureExtraction::saveCompareTable(const string& savePath)
{
	std::ofstream outStream;
	outStream.open (savePath.c_str(), std::ios::out);
	if (outStream.fail())	return false;
	
	outStream.write( (char*)(m_compareTable.datastart),
					m_compareTable.rows*m_compareTable.step[0]);
	
	outStream.close();
	
	return true;
}


bool FeatureExtraction::loadCompareTable(const string& loadPath)
{	
	ifstream inStream;
	inStream.open (loadPath.c_str(), std::ios::in);
	if (inStream.fail())	return false;
	
	// get length of file:
	inStream.seekg (0, ios::end);
	int size = inStream.tellg();
	inStream.seekg (0, ios::beg);
	
	//long size = getFeatureCount()*6*sizeof(float);
	
	char* buf = new char[size];
	
	inStream.read(buf, size);
	
	m_compareTable = Mat_<Vec6f> (size/(6*sizeof(float)),1,(Vec6f*)buf);
	m_compareTable = m_compareTable.clone();
	
	delete [] buf ;
	inStream.close();
	
	return true;
}

Mat FeatureExtraction::drawComparisons(const Mat& img)
{
	Mat out = img.clone();
	
	Colormap jetMap(Colormap::JET);
	
	if(out.empty())
		out = Mat(800,400,CV_8UC3,0.0);
	
	int divH = (int)(floor((float) out.rows / (float) m_numDivisionsH));
	int divW = (int)(floor((float) out.cols / (float) m_numDivisionsW)) ;
	
	Scalar colorStable(0.f,255.f,0.f),colorOther(255.f,0.f,0.f),colorPoint(0.f,0.f,255.f);
	
	// horizontal lines
	for (int r=1; r<m_numDivisionsH; r++) 
	{
		line(out, Point(0,r*divH), Point(out.cols,r*divH), Scalar::all(255.f), 1.f);
	}
	
	// vertical lines
	for (int c=1; c<m_numDivisionsW; c++) 
	{
		line(out, Point(c*divW,0), Point(c*divW,out.rows), Scalar::all(255.f), 1.f);
	}
	
	MatConstIterator_<Vec6f> compareIt = m_compareTable.begin(),
	compareEnd = m_compareTable.end();
	int x1,x2,y1,y2;
	
	for(;compareIt!=compareEnd;compareIt++)
	{
		const cv::Vec6f& comp = *compareIt;
		x1 = (int)floor(comp[0]*out.cols);
		y1 = (int)floor(comp[1]*out.rows);
		x2 = (int)floor(comp[2]*out.cols);
		y2 = (int)floor(comp[3]*out.rows);
		
		
		Point p1(x1,y1),p2(x2,y2);
		
		
		line(out, p1, p2, comp[4] > 0 ? jetMap.valueFor(comp[5]) : colorOther, 1.f);
		
		circle(out, p1, 1.5f, colorPoint, 1.f);
		circle(out, p2, 1.5f, colorPoint, 1.f);
		
	}
	
	return out;
	
}

void FeatureExtraction::shuffleComparisons(const cv::Mat& varImportance,const float& partDrop)
{
	Mat goodFeatureMask = Mat(1,getFeatureCount(),CV_32F,1.0);
	
	Mat goodFeatureSubMask,varImpSubMask;
	
	int dropLocal = (int)floor(m_numIntraComparisons * partDrop);
	int dropGlobal = (int)floor(m_numInterComparisons * partDrop);
	
	double minVal,maxVal;
	Point minLoc,maxLoc;
	
	if(!varImportance.empty())
	{
		Mat varTmp = varImportance.clone();
		
		minMaxLoc(varImportance,&minVal,&maxVal,&minLoc,&maxLoc);
		
		// save relative importance for each feature
		MatConstIterator_<float> importIt=varImportance.begin<float>();
		MatIterator_<Vec6f> compareIt = m_compareTable.begin(),
		compareEnd = m_compareTable.end();
		
		for(;compareIt!=compareEnd;compareIt++,importIt++)
			(*compareIt)[5] = (*importIt) / maxVal;
		
		
		// intra-Comparisons within subTiles 
		for (int r=0; r<m_numDivisionsH; r++) 
		{
			for (int c=0; c<m_numDivisionsW; c++)
			{
				
				int baseIndexForTile = (r*m_numDivisionsW + c)*m_numIntraComparisons;
				
				goodFeatureSubMask = goodFeatureMask.colRange(baseIndexForTile, baseIndexForTile+m_numIntraComparisons);
				varImpSubMask = varTmp.colRange(baseIndexForTile, baseIndexForTile+m_numIntraComparisons);
				
				minMaxLoc(varImpSubMask,&minVal,&maxVal,&minLoc,&maxLoc);
				
				// drop the n least local variables
				for(int j=0;j<dropLocal;j++)
				{
					goodFeatureSubMask.at<float>(0,minLoc.x) = -1.f;
					
					varImpSubMask.at<float>(0,minLoc.x) = -1.f;
					
					minVal=1.f;maxVal=0.f;
					
					// next maximum/minimum
					for(int i=0;i<varImpSubMask.cols;i++)
					{
						const float& val = varImpSubMask.at<float>(0,i);
						if(val >= 0.f)
						{
							if(minVal>val){ minVal = val; minLoc = Point(i,0);}
							if(maxVal<val){ maxVal = val; maxLoc = Point(i,0);}
						}
					}
					
				}
				
			}
		}
		// inter comps
		int baseIndex = (m_numDivisionsW * m_numDivisionsH)*m_numIntraComparisons;
		
		goodFeatureSubMask = goodFeatureMask.colRange(baseIndex, baseIndex+m_numInterComparisons);
		varImpSubMask = varTmp.colRange(baseIndex, baseIndex+m_numInterComparisons);
		
		minMaxLoc(varImpSubMask,&minVal,&maxVal,&minLoc,&maxLoc);
		
		// drop the n least global variables
		for(int j=0;j<dropGlobal;j++)
		{
			goodFeatureSubMask.at<float>(0,minLoc.x) = -1.f;
			
			varImpSubMask.at<float>(0,minLoc.x) = -1.f;
			
			minVal=1.f;maxVal=0.f;
			
			// next maximum/minimum
			for(int i=0;i<varImpSubMask.cols;i++)
			{
				const float& val = varImpSubMask.at<float>(0,i);
				if(val >= 0.f)
				{
					if(minVal>val){ minVal = val; minLoc = Point(i,0);}
					if(maxVal<val){ maxVal = val; maxLoc = Point(i,0);}
				}
			}
			
		}
		
		// flush
		fillCompareTable(goodFeatureMask);
		
		//cout << m_compareTable;
	}
}
	
	void FeatureExtraction::loadStandardCompareTable()
	{
		float vals []={
			0.22083266, 0.23800734, 0.046689812, 0.17458731, 0, 0.28010383,
			0.38159171, 0.15248898, 0.43400991, 0.060933538, 1, 0.29860434,
			0.22652641, 0.085320115, 0.20501755, 0.17288935, 1, 0.31710482,
			0.28682843, 0.12026137, 0.25986788, 0.24494466, 1, 0.93086654,
			0.14706005, 0.13985696, 0.44166332, 0.18113044, 1, 0.6027264,
			0.11451319, 0.076552182, 0.29083943, 0.20494381, 1, 0.34664071,
			0.44215164, 0.239353, 0.2764937, 0.093155645, 1, 0.58909446,
			0.26859578, 0.017722169, 0.42476806, 0.15139566, 1, 0.29957804,
			0.29910082, 0.098155409, 0.40869537, 0.12730838, 0, 0.28432328,
			0.12694967, 0.24920683, 0.2101166, 0.082973093, 1, 0.41609865,
			0.92253923, 0.17012507, 0.90213168, 0.11948781, 1, 0.67413175,
			0.70755267, 0.060932379, 0.82105219, 0.18506947, 0, 0.28270042,
			0.96213436, 0.20762268, 0.62695354, 0.014285614, 1, 0.36222005,
			0.73810554, 0.068315938, 0.57790565, 0.23620123, 1, 0.3800714,
			0.67038983, 0.1234922, 0.94890082, 0.02544577, 1, 0.31353456,
			0.71114838, 0.06950184, 0.90032637, 0.18447787, 1, 0.47452122,
			0.629134, 0.068713829, 0.88240993, 0.18952201, 1, 0.31807852,
			0.80093133, 0.067854241, 0.84801513, 0.13555363, 1, 0.31353456,
			0.87957495, 0.21829589, 0.65009499, 0.18310423, 0, 0.28075302,
			0.78964102, 0.034427237, 0.51241654, 0.22859849, 1, 0.36189547,
			0.063436776, 0.25318915, 0.36079502, 0.482638, 1, 0.3024992,
			0.42465216, 0.25574878, 0.07677003, 0.25292423, 1, 0.56637454,
			0.29066011, 0.44518748, 0.28843692, 0.2878392, 1, 0.50957477,
			0.012580683, 0.31921637, 0.052216653, 0.41445482, 1, 0.28691983,
			0.30032083, 0.26748875, 0.25552243, 0.36544609, 1, 1,
			0.30802545, 0.3742052, 0.45113602, 0.4848167, 1, 0.31548199,
			0.49930903, 0.43436107, 0.17956007, 0.39491051, 1, 0.283025,
			0.387707, 0.25663808, 0.23935542, 0.31202984, 1, 0.36351836,
			0.10135794, 0.27270475, 0.23502001, 0.35415962, 0, 0.29081467,
			0.44776639, 0.25384721, 0.37262473, 0.28804561, 1, 0.30347291,
			0.7817651, 0.30501315, 0.88454646, 0.38464779, 1, 0.46251217,
			0.58399159, 0.4498103, 0.94301796, 0.29147667, 1, 0.33171049,
			0.84881949, 0.49777532, 0.58945405, 0.41236567, 1, 0.27750728,
			0.9147265, 0.2720834, 0.93226767, 0.43589306, 0, 0.28107756,
			0.97887748, 0.30893493, 0.89726275, 0.38925824, 1, 0.53132099,
			0.74522525, 0.284677, 0.99148023, 0.43899179, 1, 0.4501785,
			0.58236605, 0.47746718, 0.87585616, 0.31990755, 1, 0.30639404,
			0.81529737, 0.49151704, 0.82666755, 0.38279808, 1, 0.39987019,
			0.70826238, 0.47127712, 0.82111108, 0.26390246, 1, 0.53911066,
			0.73320901, 0.42982346, 0.78876442, 0.4907093, 1, 0.29600778,
			0.2044668, 0.62332088, 0.2943584, 0.67459404, 1, 0.2937358,
			0.23035568, 0.54776722, 0.064346656, 0.74911302, 0, 0.28627068,
			0.38681582, 0.59000063, 0.39963302, 0.63470745, 0, 0.283025,
			0.41437227, 0.5639239, 0.45302519, 0.67900598, 0, 0.28919181,
			0.27994433, 0.5319429, 0.41310918, 0.61810112, 1, 0.28562155,
			0.16140881, 0.65176296, 0.21092603, 0.68044221, 0, 0.28367412,
			0.30915907, 0.50952733, 0.37191913, 0.68223941, 1, 0.29892892,
			0.12468688, 0.65367979, 0.35664731, 0.60806018, 0, 0.2839987,
			0.18541591, 0.74745321, 0.2804499, 0.64713848, 1, 0.29795519,
			0.16973171, 0.65962148, 0.38628897, 0.54684973, 1, 0.30217463,
			0.7250613, 0.51282364, 0.95829129, 0.63989985, 0, 0.28075302,
			0.50309092, 0.50530273, 0.71868813, 0.69150245, 1, 0.30671859,
			0.77513862, 0.55874002, 0.79051316, 0.71259588, 0, 0.29308665,
			0.93946064, 0.70453608, 0.9220351, 0.50740814, 1, 0.28562155,
			0.97923589, 0.55364084, 0.72862023, 0.65755373, 0, 0.29113925,
			0.85736036, 0.65317118, 0.84993243, 0.5576188, 1, 0.30801687,
			0.52995479, 0.61765039, 0.50156808, 0.54067445, 1, 0.30314833,
			0.78519237, 0.60453814, 0.62549597, 0.64457178, 0, 0.28075302,
			0.86473739, 0.61141509, 0.64810455, 0.7358954, 0, 0.27848101,
			0.73692167, 0.52470815, 0.83656335, 0.68271464, 0, 0.28821811,
			0.36516652, 0.80773163, 0.052452233, 0.92930341, 1, 0.28724441,
			0.10733309, 0.79187608, 0.35402444, 0.76181036, 1, 0.28367412,
			0.49808767, 0.93655795, 0.3623133, 0.85514283, 1, 0.2849724,
			0.011859109, 0.90433371, 0.4298709, 0.85946918, 1, 0.29341123,
			0.4339101, 0.8628003, 0.47539327, 0.93626285, 1, 0.28140214,
			0.44887096, 0.76035029, 0.49173304, 0.95616525, 1, 0.33268419,
			0.27659887, 0.98724139, 0.2941995, 0.87082022, 1, 0.30347291,
			0.42856538, 0.75038409, 0.081345253, 0.98413932, 1, 0.30736774,
			0.48310965, 0.87976813, 0.44878155, 0.75834054, 1, 0.29957804,
			0.43064466, 0.95569098, 0.11126756, 0.85888547, 1, 0.2820513,
			0.54891837, 0.82179958, 0.87358689, 0.99400711, 1, 0.3002272,
			0.80655062, 0.79011792, 0.77760011, 0.99161774, 1, 0.30931515,
			0.79948533, 0.92061329, 0.66069573, 0.7520014, 1, 0.29438493,
			0.54158133, 0.87442523, 0.56305367, 0.99565601, 1, 0.30055174,
			0.92607224, 0.80250448, 0.5309912, 0.96851724, 1, 0.2947095,
			0.80376601, 0.75601882, 0.53375101, 0.9610365, 1, 0.29957804,
			0.94978231, 0.87699986, 0.89683306, 0.84203023, 0, 0.28010383,
			0.86598951, 0.83862358, 0.78930449, 0.96542567, 0, 0.27977929,
			0.98795152, 0.81388277, 0.68557036, 0.971268, 1, 0.3024992,
			0.5993405, 0.86248261, 0.9034605, 0.9484247, 0, 0.283025,
			0.1324745, 0.4507015, 0.8318125, 0.2914055, 1, 0.30152547,
			0.16808577, 0.36362404, 0.80158353, 0.42743313, 1, 0.28789353,
			0.70489848, 0.91928101, 0.58975232, 0.054042578, 0, 0.29308665,
			0.41418639, 0.94821763, 0.36107832, 0.83651412, 1, 0.30055174,
			0.88921541, 0.10691951, 0.044474848, 0.10717042, 1, 0.27913013,
			0.35569388, 0.28393152, 0.35041031, 0.64418876, 1, 0.35118467,
			0.48458391, 0.20476526, 0.46402222, 0.53304803, 1, 0.44823107,
			0.25564525, 0.66577762, 0.53906536, 0.19516018, 1, 0.28984097,
			0.37437746, 0.2854214, 0.56416357, 0.98101062, 1, 0.3193768,
			0.99406761, 0.68495274, 0.43539286, 0.40628654, 1, 0.28270042,
			0.063316867, 0.43524235, 0.38828859, 0.31250072, 0, 0.29698148,
			0.64862531, 0.29102269, 0.71477425, 0.60412896, 1, 0.40571243,
			0.49599504, 0.96077937, 0.68157071, 0.5046373, 1, 0.28756896,
			0.94030035, 0.083138138, 0.84987926, 0.35380003, 1, 0.30477118,
			0.12403027, 0.5784331, 0.22717194, 0.4339031, 1, 0.2839987,
			0.36013687, 0.19253001, 0.15041226, 0.85856128, 1, 0.33625445,
			0.85759485, 0.46208125, 0.85516202, 0.28119385, 1, 0.69620252,
			0.46166536, 0.67740804, 0.63226128, 0.34760734, 1, 0.33106133,
			0.28975731, 0.16069153, 0.19611585, 0.84349179, 0, 0.29665691,
			0.41286573, 0.60698855, 0.31310815, 0.69396633, 1, 0.29016551,
			0.49093121, 0.37004453, 0.17148781, 0.271667, 1, 0.32392082,
			0.55689442, 0.90300179, 0.39751047, 0.28528741, 1, 0.31710482,
			0.69879991, 0.37456346, 0.66537035, 0.25550893, 0, 0.28854269,
			0.29588866, 0.49626204, 0.30059004, 0.89910191, 1, 0.29081467,
			0.81494832, 0.95224333, 0.92855322, 0.75001401, 1, 0.30736774,
			0.80225194, 0.62390602, 0.63648683, 0.36024761, 1, 0.35864979,
			0.26504967, 0.61403847, 0.29777223, 0.45958924, 1, 0.30801687,
			0.032883096, 0.14231341, 0.28211257, 0.65163195, 0, 0.29763064,
			0.18553872, 0.25483221, 0.4741115, 0.97638142, 1, 0.29698148,
			0.78951234, 0.35936889, 0.070025206, 0.83793664, 1, 0.29178837,
			0.03594353, 0.47195506, 0.42045248, 0.51798725, 1, 0.28724441,
			0.46028876, 0.29968774, 0.10610057, 0.98889184, 1, 0.43492371,
			0.72218293, 0.77077138, 0.063216038, 0.94240499, 1, 0.29406035,
			0.76880413, 0.077532262, 0.49173173, 0.41314045, 1, 0.29438493,
			0.71773642, 0.095294766, 0.083286725, 0.30897564, 1, 0.28594613,
			0.26889235, 0.67503685, 0.52805781, 0.4488422, 1, 0.30217463,
			0.72898394, 0.22716904, 0.32745603, 0.32416958, 1, 0.30834144,
			0.40952927, 0.71053976, 0.54272693, 0.13178834, 1, 0.30152547,
			0.80671251, 0.72774851, 0.64725888, 0.95796812, 1, 0.30412203,
			0.021878103, 0.89624578, 0.53736281, 0.70726097, 1, 0.28951639,
			0.98407674, 0.39053997, 0.319599, 0.47384125, 1, 0.28919181,
			0.33837253, 0.58927888, 0.40175369, 0.0062175561, 1, 0.28172672,
			0.32179227, 0.24197851, 0.5387271, 0.98806977, 0, 0.2839987,
			0.49265659, 0.47474903, 0.16576281, 0.16180223, 1, 0.28886724,
			0.73683965, 0.48348206, 0.82488447, 0.78893983, 1, 0.28821811,
			0.36243951, 0.23466852, 0.14766869, 0.37362519, 1, 0.6037001,
			0.94254661, 0.71032012, 0.84394312, 0.45724836, 1, 0.28919181,
			0.10809115, 0.91754001, 0.68727851, 0.17883974, 1, 0.30509576,
			0.92932099, 0.10011166, 0.96365434, 0.0320157, 1, 0.28919181,
			0.55774409, 0.49405137, 0.43541798, 0.19231528, 1, 0.371308,
			0.31737143, 0.63607424, 0.38501799, 0.050014354, 1, 0.28919181,
			0.14573567, 0.20991828, 0.23739442, 0.62683541, 0, 0.29665691,
			0.13403401, 0.15511806, 0.094119765, 0.9242993, 1, 0.28107756,
			0.73455209, 0.59575063, 0.72180879, 0.23773552, 1, 0.36806232,
			0.25380933, 0.48067501, 0.37365654, 0.50501293, 0, 0.28042844,
			0.015921153, 0.93247753, 0.87646645, 0.50883698, 0, 0.2839987,
			0.91469771, 0.14619674, 0.6989634, 0.76034677, 1, 0.30639404,
			0.30796611, 0.75398153, 0.015013222, 0.046067033, 0, 0.29308665,
			0.80164462, 0.33653858, 0.53298396, 0.50227052, 0, 0.28334957,
			0.95806611, 0.16014495, 0.85240477, 0.24358645, 1, 0.29957804,
			0.69177556, 0.61213511, 0.48594579, 0.30047959, 1, 0.41155469,
			0.29818439, 0.082988009, 0.36022028, 0.17623658, 1, 0.3962999,
			0.29003775, 0.034614865, 0.00097076525, 0.80459768, 1, 0.28107756,
			0.87684959, 0.32132727, 0.55825281, 0.75121289, 1, 0.31580654};
		
		int numFeatures=(sizeof(vals) / sizeof(float)) / 6;
		
		m_compareTable = Mat_<Vec6f> (numFeatures,1,(Vec6f*)vals);
		m_compareTable = m_compareTable.clone();
	}
