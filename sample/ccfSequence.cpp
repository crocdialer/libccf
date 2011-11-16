
#include <opencv2/highgui/highgui.hpp>

#include "CCF.h"
#include "sampleUtils.h"

#include <iostream>
#include <fstream>

#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;

VideoWriter imgWriter, mapWriter;

// these vars needed for calculating avg times
float avgTimeDetect=0,avgTimeGauss=0,avgTimeTraject=0;
float sumTimeDetect=0,sumTimeGauss=0,sumTimeTraject=0;

// our outPut dirs
string detectDir,gaussDir,trajectDir;

CCF_Context::DetectorEnum detector = CCF_Context::HOG_DETECTOR;

// this is our interface for the ccf-library
CCF_Context* ccfContext=NULL;

void help()
{
	cout << "\nprogram to process a image sequence using ccf\n"
	<<   "maps will be output in matlab readable textFiles\n"
	<< "Call:\n"
	<< "\n./ccfSequence [pathToSequenceRoot] [outputpath]"
	<< "\n" << endl;
	
}


Mat combineOutputMaps(const Mat& m1,const Mat& m2,const Mat& m3=Mat())
{
	assert(m1.size() == m2.size());
	if(!m3.empty()) assert(m2.size() == m3.size());
	
	int spacer = 4;
	Mat outMat = Mat(2*spacer+ 3*m1.size().height,m1.size().width,m1.type(),0.0),tmp;
	
	// First map
	Rect roi = Rect(0,0,m1.size().width,m1.size().height);
	tmp = outMat(roi);
	m1.copyTo(tmp);
	
	// Second map
	roi.y += roi.height + spacer;
	tmp = outMat(roi);
	m2.copyTo(tmp);
	
	if(!m3.empty())
	{
		// Third map
		roi.y += roi.height + spacer; 
		tmp = outMat(roi);
		m3.copyTo(tmp);
	}
	
	return outMat;
	
}

void createOutFolders(const string& resultRoot)
{
	detectDir = resultRoot+SLASH+"detect";
	gaussDir = resultRoot+SLASH+"gauss";
	trajectDir = resultRoot+SLASH+"traject";
	
#ifdef _MSC_VER // Windows
	mkdir(resultRoot.c_str());
	mkdir(detectDir.c_str());
	mkdir(gaussDir.c_str());
	mkdir(trajectDir.c_str());
	
#else // Unix
	int flags = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
	
	mkdir(resultRoot.c_str(),flags);
	mkdir(detectDir.c_str(),flags);
	mkdir(gaussDir.c_str(),flags);
	mkdir(trajectDir.c_str(),flags);
#endif
	
	
}

bool writeResults(const string& imgPath,const CCF_ResultBundle& results, const string& outDir)
{
	printf("writing results for: %s ...\n",imgPath.c_str());
	
	std::ofstream outStream;
	string outFile, txtName;
	
	
	txtName = imgPath.substr(imgPath.find_last_of(SLASH) + 1);
	txtName = txtName.substr(0,txtName.find_last_of(".")+1) + "txt";
	
	
	// detectMaps
	outFile = detectDir + SLASH + txtName;
	
	outStream.open (outFile.c_str(), std::ios::out);
	if (outStream.fail())	return false;
	
	outStream << format(results.m_detectionMap,"csv")<< endl;
	
	outStream.close();
	
	// gaussMaps
	outFile = gaussDir + SLASH + txtName;
	
	outStream.open (outFile.c_str(), std::ios::out);
	if (outStream.fail())	return false;
	
	outStream << format(results.m_confidenceMap,"csv")<< endl;
	
	outStream.close();
	
	// trajectmaps
	outFile = trajectDir + SLASH + txtName;
	
	outStream.open (outFile.c_str(), std::ios::out);
	if (outStream.fail())	return false;
	
	outStream << format(results.m_trajectorMap,"csv")<< endl;
	
	outStream.close();
	
	return true;
	
	//cout << "r (python) = " << format(ccfResults.m_trajectorMap,"python") << ";" << endl << endl;
	//cout << "r (numpy) = " << format(ccfResults.m_trajectorMap,"numpy") << ";" << endl << endl;
	
}

int main( int argc, char** argv )
{
#ifndef _MSC_VER
	chdir("../../");
#endif
	
	help();
	
	if(argc < 3)
	{
		cout << "paths not specified\n";
		return 0;
	}
	
	string rootDir = argv[1];
	string outDir = argv[2];
	
#ifdef USE_TBB
	ccfContext = new CCF_Context_TBB();
#else
	ccfContext = new CCF_Context();
#endif
	
	// detector type
	if(argc > 3)
	{
		int v;
		stringstream ss (argv[3]);
		ss>>v;
		
		detector = CCF_Context::DetectorEnum (v);
		printf("detector: %d\n",v);
	}
	
	// BB Threshold
	if(argc > 4)
	{
		float thr;
		stringstream sf (argv[4]);
		
		sf>>thr;
		
		printf("BB-Threshold: %.2f\n",thr);
		ccfContext->setBoundingBoxThreshold(thr);
		
	}
	
	FileScan fs;
	vector<string> foundFiles;
	
	fs.scanDir(rootDir,foundFiles,true);
	sort(foundFiles.begin(), foundFiles.end());
	
	printf("found %d images in '%s'\n",(int)foundFiles.size(),rootDir.c_str());
	
	createOutFolders(outDir);
	
	ccfContext->setDetector(detector);
	
	Mat detectMap,confMap,trajectMap;
	
	int count=0;
	char textBuf [1024];
	
    for(size_t i=0;i<foundFiles.size();i++)
    {
		image = imread(foundFiles[i]);
		
		Mat origImg = image.clone();
        
		ccfContext->addSingleFrame(image);
		
		CCF_ResultBundle ccfResults = ccfContext->getCurrentResultBundle();
		
		image = ccfResults.m_image;
		
		int index = i;
		
#ifndef USE_RT
		index = max((int)i - ccfContext->getFrangiFilter()->getWindowSize(),0);
#endif
		string imgPath = foundFiles[index];
		
		//if(!writeResults(imgPath,ccfResults,outDir)) break;
		
		// this section outputs coloured maps (for video generation)
		
		// generate colored	outputMaps	
		detectMap = CCF_Context::colorOutput(ccfResults.m_detectionMap,Size(320,240));
		confMap = CCF_Context::colorOutput(ccfResults.m_confidenceMap,Size(320,240));
		trajectMap = CCF_Context::colorOutput(ccfResults.m_trajectorMap,Size(320,240));
		
		string imgName = imgPath.substr(imgPath.find_last_of(SLASH) + 1);
		imgName = imgName.substr(0,imgName.find_last_of(".")+1) + "jpg";
		sprintf(textBuf, "%07d",index);
		
		// create confidence-maps outPut map
		Mat outMaps = combineOutputMaps(detectMap,confMap,trajectMap);
		
		//imwrite(detectDir + SLASH + textBuf + ".jpg", origImg);
		imwrite(gaussDir + SLASH + textBuf + ".jpg", image);
		imwrite(trajectDir + SLASH + textBuf + ".jpg", outMaps);
		
		
		// increment frame counter
		count++;
		
		if(! (i%100) )
			printf("%s -- (%d / %d)\n",(string(textBuf) + ".jpg").c_str(),(int)i,(int)foundFiles.size());
		
	}
	
	delete ccfContext;
	
    return 0;
}
