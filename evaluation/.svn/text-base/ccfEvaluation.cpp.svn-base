
#include <opencv2/highgui/highgui.hpp>

#include "CCF.h"
#include "sampleUtils.h"
#include "evaUtils.h"

#include <iostream>
#include <fstream>

#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;

// these vars needed for calculating avg times
float avgTimeDetect=0,avgTimeGauss=0,avgTimeTraject=0;
float sumTimeDetect=0,sumTimeGauss=0,sumTimeTraject=0;

// groundTruth information
map< string,vector<Rect> > groundTruth;

// start n frames earlier to adapt background
int numBufFrames = 500;

//used detector
CCF_Context::DetectorEnum detector = CCF_Context::HOG_DETECTOR;

// this is our interface for the ccf-library
CCF_Context* ccfContext=NULL;

void help()
{
	cout << "\nevaluation program for ccf\n"
	<< "Call: ./ccfEvaluation pathToSequenceRoot [detector-id]\n\n"
	<< "detector: \n\n"
	<< "0: HOG Detecor (default)\n"
	<< "1: Hue-Salience\n"
	<< "2: RandomPixel Detector\n"
	<< "\n" << endl;
	
}

int main( int argc, char** argv )
{
#ifndef _MSC_VER
	chdir("../../");
#endif
	
	help();
	
	if(argc < 2)
	{
		cout << "path to imageSeuqence needed\n";
		return 0;
	}
	
	string rootDir = argv[1];
	
	float gaussSD=0, gaussThr=0;
	
	if(argc > 2)
	{
		stringstream ss (argv[2]);
		
		ss>>gaussSD;
		
		int tmp=(int)gaussSD;// ss >> tmp;
		detector = CCF_Context::DetectorEnum (tmp);
		printf("detector: %d\n",tmp);
	}
	
	int numComp = 0,stride=0;
	
	if(argc > 3)
	{
		stringstream ss (argv[3]);

		ss>>gaussThr;
		
		//ss >> numComp;
		//printf("num mixture components: %d\n",numComp);
	}
	
	if(argc > 4)
	{
		stringstream ss (argv[4]);
		
		ss >> stride;
		//ss >> numComp;
		//printf("num mixture components: %d\n",numComp);
		printf("trajector arg(stride f. Frangi, historyLength f. Fast): %d\n",stride);
	}
	
#ifdef USE_TBB
	ccfContext = new CCF_Context_TBB();
#else
	ccfContext = new CCF_Context();
#endif
	
	// our used detector
	ccfContext->setDetector(detector);
	
	//if(numComp){ccfContext->getMixtureModel()->setNumComponents(numComp);}
	
	#ifndef USE_RT
	//if(stride){ccfContext->getFrangiFilter()->setStride(stride);}
	if(stride){ccfContext->getFrangiFilter()->setFrangiParams(.6,.6,stride);}
	
	if(false&&gaussThr)
	{ 
		ccfContext->getMixtureModel()->setParams(gaussSD, gaussThr);
		printf("mixture: sd: %.2f  -- thr: %.2f\n",gaussSD,gaussThr);
	}
	#else
	if(stride){ccfContext->getTrajector()->setHistoryLength(stride);}
	#endif
	
	FileScan fs;
	vector<string> foundFiles;
	
	printf("scanning folder ...\n\n");

	fs.scanDir(rootDir,foundFiles,true);
	sort(foundFiles.begin(), foundFiles.end());
	
	printf("found %d images in '%s'\n",(int)foundFiles.size(),rootDir.c_str());
	
	readGroundTruth("./cam3_gt.txt", groundTruth);
	
	//find first index
	int firstIndex=0;
	
	{
		string firstTagFile = "cam3_2009_05_20_04_39_20_687.jpg";
		
		vector<string>::iterator it=foundFiles.begin();
		
		for (; it!=foundFiles.end(); it++) 
		{
			string s = *it;
			
			if(s.find(firstTagFile) != string::npos)
				break;
		}
		
		firstIndex = (it - foundFiles.begin());
		
		printf("groundTruth: %d items -- first index (%s): %d \n",(int)groundTruth.size(),firstTagFile.c_str(),firstIndex);
		
		firstIndex -= numBufFrames;
	}
	
	vector<Mat> resultMats;
	
	//(numPOS,realPOS,TP) for 100 steps
	for(int m=0;m<3;m++)
		resultMats.push_back(Mat_<Vec3i>(1,100,Vec3i::all(0)));

	
	int countChecks=0,iterations=1;
	
    for(size_t i=firstIndex;i<foundFiles.size();i++)
    {
		// lastFrame
		if(foundFiles[i].find("cam3_2009_05_20_04_46_04_125.jpg") != string::npos) break;
		
		image = imread(foundFiles[i]);
		
        
		ccfContext->addSingleFrame(image);
		
		CCF_ResultBundle ccfResults = ccfContext->getCurrentResultBundle();
		
		image = ccfResults.m_image;
		
		string imgPath = foundFiles[i];
		
		#ifndef USE_RT
		int index = max((int)i - ccfContext->getFrangiFilter()->getWindowSize(),0);
		imgPath = foundFiles[index];
		#endif
		
		Mat ccfMaps[]={ccfResults.m_detectionMap,ccfResults.m_confidenceMap,ccfResults.m_trajectorMap};
		
		// groundTruth checking
		imgPath = imgPath.substr(imgPath.find_last_of(SLASH) + 1);
		map< string,vector<Rect> >::iterator it=groundTruth.find(imgPath);
		
		// check detections <-> groundtruth
		if(it != groundTruth.end())
		{
			vector<Rect>& truRects = it->second;
			
			for(int m=1;m<3;m++)
			{
				
				
				for(int i=0;i<resultMats[m].cols;i++)
				{
					float thr = (float) i / (float) resultMats[m].cols;
					vector<Rect> ourRects = ccfContext->getDectectRectangles(ccfMaps[m],image.size(),thr);
					
					resultMats[m].at<Vec3i>(0,i) += compareDetections(ourRects, truRects);
				}
				
				
			}
			if(!(countChecks %10))
				printf("cheque against groundtruth (%d / %d)\n",countChecks,(int)groundTruth.size());
			
			countChecks++;
		}
		
		CCF_Info frameInf = ccfResults.m_frameInfo;
		sumTimeDetect += frameInf.m_detectorTime;
		sumTimeGauss += frameInf.m_gaussianTime;
		sumTimeTraject += frameInf.m_trajectTime;
		
		iterations++;
	}
	
	for(int m=1;m<3;m++)
		cout << format(resultMats[m],"numpy") << "\n\n";
	
	avgTimeDetect = sumTimeDetect / (float)iterations;
	avgTimeGauss = sumTimeGauss / (float)iterations;
	avgTimeTraject = sumTimeTraject / (float)iterations;
	
	printf("avgDetectTime: %.3f -- avgGaussTime: %.3f -- avgTrajectTime: %.3f\n",
		   avgTimeDetect,avgTimeGauss,avgTimeTraject);
	
	delete ccfContext;
	
    return 0;
}
