
#include "opencv2/highgui/highgui.hpp"
#include "sampleUtils.h"
#include "CCF.h"

#include <fstream>


using namespace std;
using namespace cv;

// extracts features
FeatureExtraction extractor;

// parameters for the Randomforest
// weights for NEG, POS
const float priors[] = {1.0,1.0};

CvRTParams forestParams = CvRTParams(7,				// int _max_depth
									 10,				// int _min_sample_count
									 0,					// float _regression_accuracy
									 false,				// bool _use_surrogates
									 15,				// int _max_categories
									 priors,				// const float* _priors
									 true,				// bool _calc_var_importance
									 round(sqrt(extractor.getFeatureCount())),// int _nactive_vars
									 50,				// int max_num_of_trees_in_the_forest,
									 0.f,               // float forest_accuracy
									 CV_TERMCRIT_ITER); // int termcrit_type

BoostParams boostParams = cv::BoostParams(cv::Boost::DEFAULT, 100, 0.95, 5, false, 0 );

// the actual random forest
CvRTrees forest;

// boosting as alternative
CvBoost booster;

struct DataSet {
    int   numSamples;
    int   numPos;
    int   numNeg;
    Mat	  data;
    Mat	  responses;
    Mat   varType;
};

struct TrainResult{
	Mat var_importance;
	double train_hr;
	double test_hr;
	double fpRate;
	double fnRate;
};

bool saveHitRates(const string& savePath,const double& maxTrain,const double& maxTest,
				  const double& minFN,const double& minFP)
{
	std::ofstream outStream;
	outStream.open (savePath.c_str(), std::ios::out);
	if (outStream.fail())	return false;
	
	double outBuf[4];
	outBuf[0]=maxTrain;outBuf[1]=maxTest;
	outBuf[2]=minFN;outBuf[3]=minFP;
	
	outStream.write( (char*)outBuf,4*sizeof(double));
	
	outStream.close();
	
	return true;
}


bool loadHitRates(const string& loadPath,double* vals=NULL)
{	
	ifstream inStream;
	inStream.open (loadPath.c_str(), std::ios::in);
	if (inStream.fail())	return false;
	
	double* buf = new double[4];
	
	inStream.read((char*)buf, 4*sizeof(double));
	
	if(vals)
		memcpy(vals,buf, 4*sizeof(double));
	
	delete [] buf ;
	inStream.close();
	
	return true;
}

void loadDataSet ( string trainDir,  vector<std::string>& negData, vector<std::string>& posData, vector<std::string>& allData)
{

	printf("searching \"%s\" + \"neg\" && \"pos\" ...\n",trainDir.c_str());
	FileScan fs;
	fs.scanDir(trainDir + "/neg", negData, true);
	fs.scanDir(trainDir + "/pos", posData, true);

	allData.reserve(negData.size() + posData.size());
	allData.insert(allData.end(),negData.begin(),negData.end());
	allData.insert(allData.end(),posData.begin(),posData.end());

	printf("neg:%d  pos:%d\n",(int)negData.size(),(int)posData.size());

}

void loadDataSet ( string trainDir,  vector< pair<string,string> >& negData, vector< pair<string,string> >& posData, vector< pair<string,string> >& allData)
{
	string posFileName = trainDir+"/pos.txt";
	string negFileName = trainDir+"/neg.txt";
	if( access( posFileName.c_str(), F_OK ) != -1 and access( negFileName.c_str(), F_OK ) != -1) {
		printf("searching \"%s\" + \"pos.txt\" && \"neg.txt\" ...\n",trainDir.c_str());

		ifstream posFile;
		posFile.open(posFileName.c_str(),ios::in);

		while (!posFile.eof( ))
		{
			string file1, file2;
			getline(posFile,file1);
			getline(posFile,file2);
			posData.push_back(make_pair (file1,file2));
		}
		posFile.close( );


		ifstream negFile;
		negFile.open(negFileName.c_str(),ios::in);

		while (!negFile.eof( ))
		{
			string file1, file2;
			getline(negFile,file1);
			getline(negFile,file2);
			negData.push_back(make_pair (file1,file2));
		}
		negFile.close( );

	}



	allData.reserve(negData.size() + posData.size());
	allData.insert(allData.end(),negData.begin(),negData.end());
	allData.insert(allData.end(),posData.begin(),posData.end());

	printf("neg:%d  pos:%d\n",(int)negData.size(),(int)posData.size());

}

void evaluation(CvRTrees& forest, DataSet& data, Mat& sampleIdx, TrainResult& result)
{

	int numTrainSamples = (cv::sum( sampleIdx ))[0];

	// retrieve variable_importance
	result.var_importance = forest.get_var_importance();
//	result.var_importance = forest.get_subtree_weights();
//	cout << result.var_importance << endl;

	double min,max;
	Point minLoc,maxLoc;

	minMaxLoc(result.var_importance,&min,&max,&minLoc,&maxLoc);
//	printf("variable importance (max:%.2f%%):\n\n",max*100.f);

	// compute prediction error on train and test data
	result.train_hr = 0; result.test_hr = 0; result.fpRate = 0; result.fnRate = 0;


	Mat responses_new = Mat(data.numSamples,1,CV_32F,9.0);

	for(int i = 0; i < data.numSamples; i++ )
	{
		double r;
		Mat sample = data.data.row(i);

		// do prediction with trained forest
		r = forest.predict(sample);
		responses_new.at<float>(i,0) = r;
		float respo = data.responses.at<float>(i,0);

		// prediction correct ?
		r = fabs(r - respo) <= FLT_EPSILON ? 1 : 0;


		if( sampleIdx.at<char>(0,i) )
			result.train_hr += r;
		else
			result.test_hr += r;

		// false prediction, increase appropriate counter
		if(!r)
		{
			if(respo)
				result.fnRate += 1;
			else
				result.fpRate += 1;
		}
	}
//	cout << sampleIdx << endl;
//	cout << data.responses << endl;
//	cout << responses_new << endl;

	result.test_hr /= (double)(data.numSamples-numTrainSamples);
	result.train_hr /= (double)numTrainSamples;

	result.fpRate /= (double) data.numNeg;
	result.fnRate /= (double) data.numPos;

}


void doCrossValidation( DataSet& data, TrainResult& result)
{
	//these vars not needed - use empty Mat
	Mat varIdx, missingDataMask;

//	BoostParams forestParams = cv::BoostParams(cv::Boost::DEFAULT, 100, 0.95, 5, false, 0 );

	Mat sampleIdx;
	int nFold = 5;
	result.train_hr = 0;
	result.test_hr = 0;
	result.fpRate = 0;
	result.fnRate = 0;

//	printf( "numSamples %d", data.numSamples);

	
	// define training/test-sets within trainData
	for(int round = 0; round < nFold; round++)
	{


		//define test and trainingsset
		float partTrain = 1.0/nFold;
		sampleIdx = Mat(1,data.numSamples,CV_8U,1.0);

		int negIdx = (int)floor(partTrain*data.numNeg);
		sampleIdx.colRange(negIdx*round, negIdx*(round+1)) = 0.0;


		int posIdx = (int)floor( partTrain*data.numPos );
		sampleIdx.colRange( data.numNeg+posIdx*round, data.numNeg + posIdx*(round+1)) = 0.0;

		//int numT = (cv::sum( sampleIdx ))[0];
		//printf("sample Idx sum (trainsamples): %d\n",numT);
		
		int numTestSamples = negIdx + posIdx;
		printf("numSamples: %d -- numTrainSamples: %d -- numTestSamples: %d\n",data.numSamples, data.numSamples-numTestSamples, numTestSamples );


		//training
		forest.train(data.data, CV_ROW_SAMPLE, data.responses, varIdx, sampleIdx, data.varType, missingDataMask, forestParams);


		//evaluation
		TrainResult roundResult;
		evaluation(forest, data, sampleIdx, roundResult);

		result.fnRate 	+= roundResult.fnRate;
		result.fpRate 	+= roundResult.fpRate;
		result.test_hr 	+= roundResult.test_hr;
		result.train_hr += roundResult.train_hr;
		if( round == 0 )
			result.var_importance = roundResult.var_importance.clone();
		else
			result.var_importance += roundResult.var_importance;

		printf( "Round %d.Recognition rate: train = %.2f%%, test = %.2f%% -- overall FN = %.2f%%, FP = %.2f%%\n",
				round, roundResult.train_hr*100., roundResult.test_hr*100. ,roundResult.fnRate*100. ,roundResult.fpRate*100.);
	}
	result.fnRate 	/= nFold;
	result.fpRate 	/= nFold;
	result.test_hr 	/= nFold;
	result.train_hr /= nFold;
	result.var_importance /= nFold;
	double sum = (cv::sum(result.var_importance))[0];
	result.var_importance /= sum;

	printf( "____\nRecognition rate: train = %.2f%%, test = %.2f%% -- overall FN = %.2f%%, FP = %.2f%%\n",
			result.train_hr*100., result.test_hr*100. ,result.fnRate*100. ,result.fpRate*100.);
}

void normalValidation( DataSet& data, TrainResult& result)
{
	//these vars not needed - use empty Mat
	Mat varIdx, missingDataMask;
	
	
	
	Mat sampleIdx;

	result.train_hr = 0;
	result.test_hr = 0;
	result.fpRate = 0;
	result.fnRate = 0;
	
	//	printf( "numSamples %d", data.numSamples);
	
	//CvBoostTree boost;
	
	//define test and trainingsset
	float partTrain = 1.0/8.0;
	sampleIdx = Mat(1,data.numSamples,CV_8U,1.0);
	
	int negIdx = (int)floor(partTrain*data.numNeg);
	sampleIdx.colRange(negIdx*5, negIdx*6) = 0.0;
	
	
	int posIdx = (int)floor( partTrain*data.numPos );
	sampleIdx.colRange( data.numNeg+posIdx*5, data.numNeg + posIdx*6) = 0.0;
	
	//int numT = (cv::sum( sampleIdx ))[0];
	//printf("sample Idx sum (trainsamples): %d\n",numT);
	
	int numTestSamples = negIdx + posIdx;
	printf("numSamples: %d -- numTrainSamples: %d -- numTestSamples: %d\n",data.numSamples, data.numSamples-numTestSamples, numTestSamples );
	
	
	//training
	forest.train(data.data, CV_ROW_SAMPLE, data.responses, varIdx, sampleIdx, data.varType, missingDataMask, forestParams);
	
	//booster.train(data.data, CV_ROW_SAMPLE, data.responses, varIdx, sampleIdx, data.varType, missingDataMask, boostParams);
	
	//evaluation
	evaluation(forest, data, sampleIdx, result);
	
	
	double sum = (cv::sum(result.var_importance))[0];
	result.var_importance /= sum;
	
	printf( "____\nRecognition rate: train = %.2f%%, test = %.2f%% -- overall FN = %.2f%%, FP = %.2f%%\n",
		   result.train_hr*100., result.test_hr*100. ,result.fnRate*100. ,result.fpRate*100.);
}

int main (int argc, const char * argv[]) 
{
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	
//	vector< pair<string,string> > negData,posData,allData;
	vector< string > negData,posData,allData;
	char workingDir[512];
	getcwd(workingDir, 512);

	string trainDir;
	if (argc < 2)
		trainDir = workingDir;
	else
		trainDir = argv[1];
	
	int numIntra,numInter,numH=4,numW=2;
	
	if(argc > 3)
	{
		stringstream ss (argv[2]);
		
		int numFeats;
		ss>>numFeats;
		
		float localRatio;
		stringstream sf (argv[3]);
		
		sf>>localRatio;
		
		printf("%d -- %.2f\n",numFeats,localRatio);
		
		numIntra = numFeats*localRatio / float(numH*numW) ;
		numInter = numFeats-numIntra*numH*numW ;
		
		extractor = FeatureExtraction(numH,numW,numIntra,numInter);
		
		if(argc < 5) printf("\nset features: numH = %d, numW = %d, numIntra = %d, numInter = %d\n",numH,numW,numIntra,numInter);
		
		forestParams.nactive_vars = round(sqrt(extractor.getFeatureCount()));
		
	}
	
	if(argc > 5)
	{
		stringstream ss (argv[4]);
		
		ss>>numH;
		
		stringstream sf (argv[5]);
		sf>>numW;
		
		extractor = FeatureExtraction(numH,numW,numIntra,numInter);
		printf("\nset features: numH = %d, numW = %d, numIntra = %d, numInter = %d\n",numH,numW,numIntra,numInter);
		
	}
	
	// disable hue-comparisons here
	//extractor.setUseHueComparisons(false);
	
	// load comparetable from disk
	//extractor.loadCompareTable (trainDir+"/compareTable");
	
	loadDataSet( trainDir, negData,posData,allData);

    printf("\nPersona Trainer -> %s\n",asctime(timeinfo));
	
	// compute prediction error on train and test data
	double maxTrainHR = 0,maxTestHR = 0,minFNRate = 1,minFPRate = 1;
	//double valAr[4];
	
	RNG randomGen(rawtime);
	
	DataSet data;
	data.numNeg = negData.size();
	data.numPos = posData.size();
	data.numSamples = data.numNeg + data.numPos;

	if(!data.numSamples)
	{
		printf("no images found :(\n");
		return 1;
	}
	
	// set type of feature variables to categorical
	data.varType = Mat(extractor.getFeatureCount()+1, 1, CV_8UC1, CV_VAR_CATEGORICAL);
	
	// label-responses for data
	data.responses = Mat(data.numSamples,1,CV_32F,0.0);
	data.responses.rowRange(Range(data.numNeg,data.numSamples)) = Scalar::all(1.0);
	
	// a matrix holding all feature-vectors for our samples
	data.data = Mat(data.numSamples,extractor.getFeatureCount(),CV_32F,1.0);
	
	
	double elapsedTime;
	// fitting for nicta dataset
	Rect dRect = Rect(14,5,36,72 );
	
	// my old time-helper pal
	TimerUtil timer;
	
	// construct posMeanImg
	Mat posMeanImg; 
	
	// for all samples, create featureVectors and put into trainData (holds training+test sets)
	Mat featureVec,row,img,compImg;
	MatConstIterator_<float> featureIt,featureEnd;
	MatIterator_<float> rowIt,rowEnd;
	
	int iterCount=1;
	
	TrainResult result;
	result.fnRate =0; result.fpRate =0; result.test_hr =0; result.train_hr =0;

	while (result.test_hr < 0.95)
	{
		// load compareTable (more than 1 process working on this)
		//extractor.loadCompareTable (trainDir+"/compareTable");
		
		compImg = extractor.drawComparisons();
		imwrite(trainDir+"/compImg_raw.jpg", compImg);
		
		posMeanImg = Mat(dRect.size(),CV_32FC3,0.0);
		
		timer.reset();
		
		printf("starting extraction of features -> iteration %d\n",iterCount++);
		Mat data2 = Mat(data.numSamples,1,CV_32F,1.0);
		// extract features for all samples
		for (int i=0; i<data.numSamples; i++)
		{
			if(! (i%1000) ) 
			{
				printf("%d / %d\n",i,data.numSamples);
			}
			
			img = imread(allData[i], 1);
			extractor.setImage(img);
			
			if(i>(int)negData.size())
			{ 
				img.convertTo(img, CV_32FC3);
				posMeanImg += img(dRect);
			}
			
			featureVec = extractor.createFeatureVector(dRect);
			row = data.data.row(i);
			
			featureVec.copyTo(row);
		}
		printf("\n");
		
		elapsedTime = timer.getElapsedTime();
		
		printf("extraction of %d features in %d images took %.4f ms\n",extractor.getFeatureCount(),data.numSamples,elapsedTime);
		
		posMeanImg /= (float) posData.size();
		posMeanImg.convertTo(posMeanImg, CV_8UC3);
		Mat resized;
		cv::resize (posMeanImg, resized,Size(256,512));
		
		// draw comparsisons to image
		compImg = extractor.drawComparisons(resized);
		
		
		// Random Forest Classifier
		//doCrossValidation( data, result);
		normalValidation(data, result);
		
		if( ( (result.test_hr+result.train_hr)/2.0 > (maxTestHR+maxTrainHR)/2.0) )
		  // && (result.fnRate<minFNRate) )
		{
			printf("\nResult improved !! - Saving compareTable+classifier\n\n");
			
			if(minFNRate > result.fnRate) minFNRate = result.fnRate;
			if(minFPRate > result.fpRate) minFPRate = result.fpRate;
			if(maxTestHR < result.test_hr) maxTestHR = result.test_hr;
			if(maxTrainHR < result.train_hr) maxTrainHR = result.train_hr;
			
			// Save new hitRates
			//saveHitRates(trainDir + "/hitRates",maxTrainHR,maxTestHR,minFNRate,minFPRate);
			
			imwrite(trainDir+"/compImg.jpg", compImg);
			
			// Save classifier to disk
			string savePath = trainDir+"/RT_PersonClassifier.yaml";
			forest.save(savePath.c_str(), "RT_PersonClassifier");
			
			// Save compareTable
			extractor.saveCompareTable(trainDir+"/compareTable");
		}
		
		if(minFNRate > result.fnRate) minFNRate = result.fnRate;
		if(minFPRate > result.fpRate) minFPRate = result.fpRate;
		if(maxTestHR < result.test_hr) maxTestHR = result.test_hr;
		if(maxTrainHR < result.train_hr) maxTrainHR = result.train_hr;
		
	
		// fraction of features to drop when shuffling
		float partDrop = .5;
		
		printf("num Features: %d -- dropping: %d\n\n",extractor.getFeatureCount(),
			   (int)(partDrop*extractor.getFeatureCount()) );
		
		// drop one third of comparison
		extractor.shuffleComparisons(result.var_importance,partDrop);
	}
	
	

    return 0;
}
