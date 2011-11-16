
#include <opencv2/highgui/highgui.hpp>

#include "CCF.h"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;

//marking within image
Point origin;
Rect selection;
vector<Rect> tagRects;

// control vars
bool mouseSelect=false,showGrid=false,tagging=false;
bool drawText=true,overlayMap=false;

// these vars needed for calculating avg times
float avgTimeDetect=0,avgTimeGauss=0,avgTimeTraject=0;
float sumTimeDetect=0,sumTimeGauss=0,sumTimeTraject=0;

// calc avg every n frames
int avgCount=25;

int outWidth = 1024;

// this is our interface for the ccf-library
CCF_Context* ccfContext=NULL;

void help()
{
	cout << "\nThis is a demo that demonstrates use of the ccf-library\n"
	<<   "This reads from video camera (0 by default, or the camera number the user enters\n"
	<< "Call:\n"
	<< "\n./ccfCam [camera number]"
	<< "\n" << endl;
	
	cout << "\n\nHot keys: \n"
	"\tESC - quit the program\n"
	"\td - change detector type (HOG,SimpleSalience,RandomPixels)\n"
	"\tg - show / hide grid\n"
	"\to - overlay confidence map\n"
	"\tt - enable / disable tagging - will adapt the detection grid\n"
	"\t(mark size of smallest and largest person in input)\n"<< endl;
	
	#ifdef USE_TBB
	cout << "\ti - enable / disable use of Intel TBB\n";
	#endif
	
	cout << "\tp - print frame infos\n";
}

void onMouse( int event, int x, int y, int, void* )
{
	if(!tagging) return;
	
    if( mouseSelect )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
		
        selection &= Rect(0, 0, image.cols, image.rows);
    }
	
    switch( event )
    {
		case CV_EVENT_LBUTTONDOWN:
			origin = Point(x,y);
			selection = Rect(x,y,0,0);
			mouseSelect = true;
			break;
		case CV_EVENT_LBUTTONUP:
			mouseSelect = false;
			if( selection.width > 0 && selection.height > 0 )
			{
				tagRects.push_back(selection);
				ccfContext->setGrid(image.size(),tagRects);
				
			}
			break;
    }
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

int main( int argc, char** argv )
{
#ifndef _MSC_VER
	chdir("../../");
#endif
	
#ifdef USE_TBB
	bool useTBB = true;
	ccfContext = new CCF_Context_TBB();
#else
	ccfContext = new CCF_Context();
#endif
	
	ccfContext->setDetector(CCF_Context::FRESH_DETECTOR);
	
    VideoCapture cap;
    Rect trackWindow;
	
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);
	
    if( !cap.isOpened() )
    {
    	help();
        cout << "***Could not initialize capturing...***\n";
        return 0;
    }
	
    help();
	
    namedWindow( "CCF Demo", 1 );
	namedWindow( "Confidence Maps", 1 );
	
    setMouseCallback( "CCF Demo", onMouse, 0 );

	Mat detectMap,confMap,trajectMap;
	
	int count=0;
	
    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;
		
		float ratio = outWidth / (float)frame.cols;
        resize(frame,image, Size(),ratio,ratio);
        
		ccfContext->addSingleFrame(image);
		
		CCF_ResultBundle ccfResults = ccfContext->getCurrentResultBundle();
		
		image = ccfResults.m_image;

		// generate colored	outputMaps		 
		detectMap = CCF_Context::colorOutput(ccfResults.m_detectionMap,Size(320,240));
		confMap = CCF_Context::colorOutput(ccfResults.m_confidenceMap,Size(320,240));
		trajectMap = CCF_Context::colorOutput(ccfResults.m_trajectorMap,Size(320,240));
		
		//draw time labels into maps
		if(drawText)
		{
			CCF_Info inf = ccfResults.m_frameInfo;
			
			sumTimeDetect += inf.m_detectorTime;
			sumTimeGauss += inf.m_gaussianTime;
			sumTimeTraject += inf.m_trajectTime;
			
			if(! (count% avgCount) )
			{
				avgTimeDetect = sumTimeDetect / (float) avgCount;
				avgTimeGauss = sumTimeGauss / (float) avgCount;
				avgTimeTraject = sumTimeTraject / (float) avgCount;
				
				sumTimeDetect=sumTimeGauss=sumTimeTraject=0;
				count = 0;
			}
			
			int font = FONT_HERSHEY_DUPLEX;
			double scale = .6, xs_scale = .4;
			Scalar color = Scalar::all(255);
			Point pos = Point(8,20);
			Point offSet = Point(0,20);
			
			char buf[64];
			sprintf(buf, "%.3f ms",inf.m_detectorTime);
			putText(detectMap, string(buf), pos, font,scale,color);
			sprintf(buf, "~ %.3f ms",avgTimeDetect);
			putText(detectMap, string(buf), pos+offSet, font,xs_scale,color);
			sprintf(buf,"input: %d x %d",image.size().width,image.size().height);
			putText(detectMap, string(buf), pos+offSet*2, font,xs_scale,color);
			
			sprintf(buf, "%.3f ms",inf.m_gaussianTime);
			putText(confMap, string(buf), pos, font,scale,color);
			sprintf(buf, "~ %.3f ms",avgTimeGauss);
			putText(confMap, string(buf), pos+offSet, font,xs_scale,color);
			
			sprintf(buf, "%.3f ms",inf.m_trajectTime);
			putText(trajectMap, string(buf), pos, font,scale,color);
			sprintf(buf, "~ %.3f ms",avgTimeTraject);
			putText(trajectMap, string(buf), pos+offSet, font,xs_scale,color);
		
		}
			
		// create confidence-maps outPut map
		Mat outMaps = combineOutputMaps(detectMap,confMap,trajectMap);

		// draw selection if any
		if( mouseSelect )
		{
			Mat selectMat = image(selection);
			bitwise_not(selectMat,selectMat);
		}
		
		// update windows
        imshow( "CCF Demo", image );
        imshow( "Confidence Maps", outMaps );
		
		// increment frame counter
		count++;
		
        char c = (char)waitKey(10);
        if( c == 27 || c == 'q')
            break;
		
		CCF_Context::DetectorEnum d = (CCF_Context::DetectorEnum)(((int)ccfContext->getDetectorType() + 1) % 3);
        switch(c)
        {
			case 'd':
				ccfContext->setDetector(d);	
			break;
			
			case 'g':
				showGrid = !showGrid;
				
				ccfContext->setDrawGrid(showGrid);
				
				break;
				
			case 't':
				
				tagRects.clear();
				tagging = !tagging;
				printf("mouse tagging %s\n",tagging? "enabled":"disabled");
				
				break;
			
			#ifdef USE_TBB
			case 'i':
				
				useTBB = !useTBB;
				delete ccfContext;
				
				if(useTBB) ccfContext = new CCF_Context_TBB();
				else ccfContext = new CCF_Context();
				
				ccfContext->setDetector(CCF_Context::FRESH_DETECTOR);
				
				printf("TBB %s\n",useTBB? "enabled":"disabled");
				
				break;
			#endif
				
			case 'p':
				
				drawText = !drawText;
				
				break;
				
			case 'o':
				
				overlayMap = !overlayMap;
				ccfContext->setDrawOverlay(overlayMap);
				
				break;
				
			default:
				break;
        }
    }
	
	delete ccfContext;
	
    return 0;
}
