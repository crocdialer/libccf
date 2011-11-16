/*
 *  evaUtil.h
 *  CCF_Library
 *
 *  Created by Fabian on 2/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef EVAUTILS_H
#define EVAUTILS_H

#include "opencv2/opencv.hpp"
#include <fstream>

using namespace std;
using namespace cv;

void stringExplode(const string& str, const string& separator, vector<string>& results)
{
	results.clear();
	string subStr=str;
	
    size_t found;
    found = subStr.find_first_of(separator);
	
    while(found != std::string::npos)
	{
        if(found > 0)
		{
            results.push_back(subStr.substr(0,found));
        }
        subStr = subStr.substr(found+1);
        found = subStr.find_first_of(separator);
    }
    if(subStr.length() > 0)
	{
        results.push_back(subStr);
    }
};


bool readGroundTruth(const string& p,map< string,vector<Rect> >& outMap)
{
	
	ifstream inStream;
	inStream.open (p.c_str(), std::ios::in);
	if (inStream.fail())	return false;
	
	char buf [1024];
	
	string line ="foo",fileName,rectString;
	
	
	vector<string> subStrs;
	
	vector<Rect> foundRects;
	
	while(!line.empty())
	{
		foundRects.clear();
		
		inStream.getline(buf, 1024);
		line = buf;
		stringExplode(line," ",subStrs);
		
		fileName = line.substr(0,line.find_first_of(" ") );
		fileName = fileName.substr(fileName.find_last_of("\\") + 1);
		
		rectString = line.substr(line.find_first_of(" ") +1);
		
		stringExplode(rectString,"[",subStrs);
		
		int x,y,width,height;
		
		for (size_t i=0; i<subStrs.size(); i++) 
		{
			subStrs[i] = subStrs[i].substr(0,subStrs[i].size()-2);
			
			stringstream sstream (subStrs[i]);
			sstream >> x;sstream >> y;
			sstream >> width; sstream >> height;
			
			foundRects.push_back(Rect(x,y,width,height));
			
		}
		
		// insert entry into map
		outMap[fileName] = foundRects;
		
	}
	
	
	inStream.close();
	
	
	return true;
};

Vec3i compareDetections(const vector<Rect>& det,const vector<Rect>& truth)
{
	vector<Rect> detCpy = det;
	
	Vec3i ret (detCpy.size(),truth.size(),0);//(numPOS,realPOS,TP)
	
	for (size_t j=0; j<truth.size(); j++)
	{
		const Rect& trueRect = truth[j];
		
		for (size_t i=0; i<detCpy.size(); i++)
		{
			Rect& detRect = detCpy[i];
			
			Rect overLap = detRect & trueRect;
			
			// Rects are not related
			if(overLap.area() == 0) continue;
				
			// check overlap (pascal criteria)
			if( (detRect.area()+trueRect.area()-overLap.area()) /
			   overLap.area() >= 0.5)
			{
				ret[2]++;
				
				// remove from hitlist
				detRect = Rect();
				
				break;
			}
		} 
	}
	
	return ret;
};

#endif // EVAUTILS_H