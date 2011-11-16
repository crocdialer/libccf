/***************************************************************************                                              *
 *   GaussianMixture class - Fabian Schmidt                                      *
 *
 *   Author of MATLAB code: Seth Benton                                    *
 *   http://www.sethbenton.com/mixture_of_gaussians.html                   *
 *                                                                         *
 *   This class implements the mixture of Gaussians algorithm for          *
 *   background subtraction.                                               *
 ***************************************************************************/

#include "GaussianMixture.h"

using namespace std;
using namespace cv;

GaussianMixture::GaussianMixture(const Size& s):m_size(s),m_initMean(true),m_maxValue(1.0)
{
	//initialize Gaussian mixture model
	m_numMixtureComponents = 2;
	
	//positive deviation threshold
	m_devThresh	=	3.5;
	
	//learning rate (between 0 and 1) (from paper 0.01)
	m_learningRate		=	0.002;
	
	//foreground threshold (0.25 or 0.75 in paper)
	//m_fgThresh		=	0.25;	
	
	//initial standard deviation (for new components) var = 36 in paper
	m_initialSD	=	6;
	
	//initial p variable (used to update mean and sd)
	m_p = m_learningRate * m_numMixtureComponents;
	
	initArrays();
	
	// alternative implementation
	m_backgroundModel = new BackgroundSubtractorMOG(1.0 / m_learningRate, //history
													m_numMixtureComponents, 
													0.85); // backgroundratio
	//15); // noise sigma
	
	m_fgMask = Mat(m_size,CV_8UC1);
	
}

GaussianMixture::~GaussianMixture()
{
	
	
}

void GaussianMixture::initArrays()
{
	m_rank.clear();
	m_pixDiff.clear();
	m_weights.clear();
	m_pixSD.clear();
	m_pixMean.clear();
	
	RNG randomGen;
	
	for(int i=0;i<m_numMixtureComponents;i++)
	{
		
		m_rank.push_back( 0.0 );
		
		m_pixDiff.push_back( Mat_<double> (m_size, 0.0)); 
		
		//initialize weights array, uniformly dist
		m_weights.push_back( Mat_<double> (m_size,1.0/m_numMixtureComponents));
		
		m_pixSD.push_back( Mat_<double> (m_size,m_initialSD)); 
		
		m_pixMean.push_back( Mat_<double> (m_size,0.0));
		
	}
	
}

inline void GaussianMixture::setMean(const Mat_<double>& newMean)
{
	for(int i=0;i<m_numMixtureComponents;i++)
		m_pixMean[i] = newMean.clone() ;
}

void GaussianMixture::resetBackground()
{
	m_initMean = true;
	m_maxValue = 1.0;
	initArrays();
	
	//m_backgroundModel->initialize(m_size,CV_8UC1);
	delete m_backgroundModel;
	m_backgroundModel = new BackgroundSubtractorMOG(1.0 / m_learningRate, //history
													m_numMixtureComponents, 
													0.85); // backgroundratio
}

void GaussianMixture::filter(const Mat_<double>& src,Mat_<double>& dst)
{
	assert(src.size() == m_size);
	
	//if(dst.size() != src.size() || dst.type() != src.type())
	//	dst.create( src.size(),src.type() );
	int* rankIndex = new int[m_numMixtureComponents];
	
	//init  mean with first frame
	if (m_initMean) 
	{
		setMean(src);
		m_initMean = false;
	}
	
	//calculate difference of pixel values from mean
	for (int i=0; i<m_numMixtureComponents; i++)
		//no absolute value -> we are only interested in positive side
		m_pixDiff[i] = src - m_pixMean[i];
	
	//update gaussian components for each pixel
	for (int i=0; i < m_size.height; i++) 
	{
		for (int j=0; j<m_size.width; j++) 
		{
			bool match = false;
			for (int k=0; k<m_numMixtureComponents; k++) 
			{
				
				//pixel matches component
				if ( abs(m_pixDiff[k][i][j]) <= m_devThresh * m_pixSD[k][i][j]) 
				{
					
					//variable to signal component match
					match = true;	
					
					//update weights, mean, sd, p
					m_weights[k][i][j] = (1-m_learningRate)*m_weights[k][i][j] + m_learningRate;
					m_p = m_learningRate/m_weights[k][i][j];
					m_pixMean[k][i][j] = (1 - m_p) * m_pixMean[k][i][j] + m_p * src [i][j];
					m_pixSD[k][i][j] = sqrt((1 - m_p)*m_pixSD[k][i][j]*m_pixSD[k][i][j] + m_p*(src[i][j]-m_pixMean[k][i][j])*(src[i][j]-m_pixMean[k][i][j]));
				}
				
				//pixel doesn't match component
				else 
				{	
					//weight slighly decreases
					m_weights[k][i][j] = (1 - m_learningRate) * m_weights[k][i][j];
				}
			}
			
			double sumWeights = 0;
			for (int k=0; k<m_numMixtureComponents; k++)
				sumWeights += m_weights[k][i][j];
			
			for (int k=0; k<m_numMixtureComponents; k++) 
			{
				m_weights[k][i][j] /=sumWeights;
			}
			
			//if no components match, create new component
			if (!match) 
			{
				double minWeight = m_weights[0][i][j];
				int minWeightIndex = 0;
				for (int k=0; k<m_numMixtureComponents; k++)
					if(minWeight>m_weights[k][i][j]) 
					{
						minWeight = m_weights[k][i][j];
						minWeightIndex = k;
					}
				m_pixMean [minWeightIndex][i][j] = src[i][j];
				m_pixSD [minWeightIndex][i][j] = m_initialSD;
			}
			
			//calculate component rank
			for (int k=0; k<m_numMixtureComponents; k++) 
			{
				m_rank[k] = m_weights[k][i][j] / m_pixSD[k][i][j];
			}
			
			for (int k=0; k<m_numMixtureComponents; k++)
				rankIndex[k] = k;
			
			//sort rank values
			double rankTemp;
			int rankIndexTemp;
			for (int k=1; k<m_numMixtureComponents; k++) 
			{
				for (int m=0; m<k; m++) 
				{
					if (m_rank[k] > m_rank[m]) 
					{
						//swap max values
						rankTemp = m_rank[m];
						m_rank[m] = m_rank[k];
						m_rank[k] = rankTemp;
						
						//swap max index values
						rankIndexTemp = rankIndex[m];
						rankIndex[m] = rankIndex[k];
						rankIndex[k] = rankIndexTemp;
					}
				}
			}
			//BG is the dominant mode
			//BG is the mode with the lowest mean
			//calculate the Mahanalobis distance to that dominant mode
			int k=0;	
			
			dst[i][j] = m_pixDiff[rankIndex[k]][i][j] / m_pixSD[rankIndex[k]][i][j];
		}
	}
	
	// remove negative values
	max(dst,0.0,dst);
	
	/*
	 //TODO: test if this is usefull
	 double min,max;
	 minMaxLoc(dst,&min,&max);
	 m_maxValue = std::max(max,m_maxValue);	
	 
	 if(m_maxValue>1) 
	 dst /= m_maxValue;
	 
	 // calm down slowly
	 m_maxValue *= (1-m_learningRate);
	 
	 */
	delete[] rankIndex;
}

void GaussianMixture::filterAlt(const Mat& src,Mat& dst)
{
	Mat tmp;
	tmp = src * 255.0;
	tmp.convertTo(tmp,CV_8UC1);
	
	//TODO: test opencv implementation of mixture of gaussians here
	(*m_backgroundModel)(tmp,m_fgMask,m_learningRate);
	m_fgMask.convertTo(tmp, src.type());
	tmp /= 255.0;
	multiply(src, tmp, dst);
	
	double maxConf;
	minMaxLoc(dst, NULL, &maxConf);
	GaussianBlur(dst, dst, Size(5,5),0);
	normalize(dst, dst, 0, maxConf, CV_MINMAX);
}