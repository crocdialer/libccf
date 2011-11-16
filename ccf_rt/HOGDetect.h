/***************************************************************************
 *   Extended 2010/08/25 by Yixuan Yang from modiified cvaux.hpp and       *
 *   cvhog.cpp from Severin Stalder.                                       *
 *                                                                         *
 *   This HOGDetect.cpp is for HOG (Histogram-of-Oriented Gradients        *
 *   descriptor and object detection in CCF program.                       *
 ***************************************************************************/


/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef HOGDetectCCF_H
#define HOGDetectCCF_H

#ifdef HAVE_CONFIG_H
//#undef HAVE_CONFIG_H
#endif


#include "opencv2/opencv.hpp"

using namespace cv;


/****************************************************************************************\
*            HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector        *
\****************************************************************************************/

class HOGDescriptorCCF
{
public:
    enum { L2Hys=0 };

    HOGDescriptorCCF() : winSize(64,128), blockSize(16,16), blockStride(8,8),
        cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(L2Hys), L2HysThreshold(0.2), gammaCorrection(true)
    {};

    HOGDescriptorCCF(Size _winSize, Size _blockSize, Size _blockStride,
        Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
        int _histogramNormType=L2Hys, double _L2HysThreshold=0.2, bool _gammaCorrection=false)
        : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
        nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
        histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
        gammaCorrection(_gammaCorrection)
    {};

    HOGDescriptorCCF(const String& filename)
    {
        load(filename);
    };

    virtual ~HOGDescriptorCCF() {};

    size_t getDescriptorSize() const;
    bool checkDetectorSize() const;
    double getWinSigma() const;

    virtual void setSVMDetector(const vector<float>& _svmdetector);

    virtual bool load(const String& filename, const String& objname=String());
    virtual void save(const String& filename, const String& objname=String()) const;

    virtual void compute(const Mat& img,
                         vector<float>& descriptors,
                         Size winStride=Size(), Size padding=Size(),
                         const vector<Point>& locations=vector<Point>()) const;

    virtual void detect(const Mat& img, vector<Point>& foundLocations,
                        vector<double>& score, double hitThreshold=0, Size winStride=Size(), 
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const; //vector<int> score!!!

//virtual void detect(const Mat& img, vector<Point>& hits, vector<double>& score, double hitThreshold,
//    Size winStride, Size padding, const vector<Point>& locations) const;

    //new
	//virtual void detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<int>& score, 
 //                                 double hitThreshold=0, Size winStride=Size(),
 //                                 Size padding=Size(), double scale=1.05,
 //                                 int groupThreshold=2) const;

	virtual void detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<double>& score, 
    double hitThreshold, Size winStride, Size padding,
    double scale0, int groupThreshold=2) const;

	////orig
	//    virtual void detectMultiScale(const Mat& img, vector<Rect>& foundLocations, 
 //                                 double hitThreshold=0, Size winStride=Size(),
 //                                 Size padding=Size(), double scale=1.05,
 //                                 int groupThreshold=2) const;

    virtual void computeGradient(const Mat& img, Mat& grad, Mat& angleOfs,
                                 Size paddingTL=Size(), Size paddingBR=Size()) const;

    static vector<float> getDefaultPeopleDetector();

//private:
	
    Size winSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    vector<float> svmDetector;
};


#endif //HOGDetectCCF_H
