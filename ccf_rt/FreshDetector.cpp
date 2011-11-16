#include "FreshDetector.h"
#include "CCF_Context.h"

using namespace cv;
using namespace std;

FreshDetector::FreshDetector(const string& compareTable, const string& classifier)
{
	m_extractor = new FeatureExtraction();
	
	if(!(compareTable.empty()))
	{
		m_extractor->loadCompareTable(compareTable);
	}
	
	if(!classifier.empty())
	{
		loadClassifier(classifier);
	}
}

FreshDetector::~FreshDetector()
{
	delete m_extractor;
	
}

void FreshDetector::detectForScale(const Rect& stripeRect, vector<double>& score, const Size& winStride)
{	
	if(m_imageSize == Size()) return;
	
	int xPos = 0, yPos = stripeRect.y;
	int height=stripeRect.height,width=stripeRect.height / 2;
	
	Mat pixTab = m_extractor->buildPixelTable(Rect(0,0,width,height));
	Mat featureVec;
	
	while (xPos < m_imageSize.width - width) 
	{
		// adjust size of detectWindow
		Rect detectRect(xPos,yPos,width,height );
		
		featureVec = (m_extractor->createFeatureVector(detectRect,&pixTab));
	
		// This method returns the fraction of tree votes for class 1 (Person). 
		// The method works for binary classification problems only
		score.push_back(m_forest.predict_prob(featureVec));
		
		xPos += winStride.width ;
	}

	//printf("createFeatureVector: %.4f ms -- forest.predict: %.4f ms\n",t1,t2);
}