#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "unscented_kalman_filter.h"
#include "RootMeanSquareError.h"

using namespace std;

double       k = 0.0;		//! loop count
const double T = 200.0;          //! loop limit
const int iterate = 1;
//----------------------------
// Process Equation
//! x		: 状態ベクトル
//! xpre	: 1時刻前の状態ベクトル
//! input	: 制御入力
void process(cv::Mat &x, const cv::Mat &xpre, const double &input)
{
    // x.at<double>(0, 0) =  0.2*xpre.at<double>(0,0) 
	//   + 25.0*(xpre.at<double>(0,0) / (1.0 + (xpre.at<double>(0,0)*xpre.at<double>(0,0)))) 
	//   +  8.0 * cos(1.2*k);
  // x.at<double>(0, 0) =  xpre.at<double>(0,0) + sin(xpre.at<double>(0,0)/2.0) + cos(xpre.at<double>(0,0));
}

//-------------------------
// Observation Equation
//! z : 観測値
//! x : 状態ベクトル
void observation(cv::Mat &z, const cv::Mat &x)
{
  z.at<double>(0, 0) = pow(x.at<double>(0, 0) * x.at<double>(0, 0),2.0)/20.0  ;
  //  z.at<double>(0, 0) = pow(x.at<double>(0, 0) * x.at<double>(0, 0),4.0);
}


int main(void)
{
  ofstream outputresult;        // x, y, est
  outputresult.open("result0.dat", ios::out);
  if (!outputresult.is_open()){ std::cout << "open result output failed" << endl; return -1; }
  
  const int state_dimension = 1;
  const int output_dimension = 1;
  cv::Mat x0 = (cv::Mat_<double>(state_dimension, 1) << 0.0);
  cv::Mat p0 = (cv::Mat_<double>(state_dimension, 1) << 1.0);

  cv::Mat ProcessNoiseCov  = (cv::Mat_<double>(state_dimension, 1) << 1.0);
  cv::Mat ProcessNoiseMean = (cv::Mat_<double>(state_dimension, 1) << 0.0);
  cv::Mat ObsNoiseCov  = (cv::Mat_<double>(output_dimension, 1) << 10.0);
  cv::Mat ObsNoiseMean = (cv::Mat_<double>(output_dimension, 1) << 0);
  
  cv::Mat state             = cv::Mat::zeros(state_dimension,  1, CV_64F); /* (x) */
  cv::Mat last_state        = cv::Mat::zeros(state_dimension,  1, CV_64F); /* (x) */
  cv::Mat processNoise      = cv::Mat::zeros(state_dimension,  1, CV_64F);
  cv::Mat measurement       = cv::Mat::zeros(output_dimension, 1, CV_64F);
  cv::Mat measurementNoise  = cv::Mat::zeros(output_dimension, 1, CV_64F);
  cv::Mat first_sensor      = cv::Mat::zeros(output_dimension, 1, CV_64F);

  static random_device rdev;
  static mt19937 engine(rdev());
  normal_distribution<> processNoiseGen(ProcessNoiseMean.at<double>(0, 0)
										, sqrt(ProcessNoiseCov.at<double>(0, 0)));
  normal_distribution<> obsNoiseGen(ObsNoiseMean.at<double>(0, 0)
									, sqrt(ObsNoiseCov.at<double>(0, 0)));
    
  double rmse_average = 0;

  for(int loop = 0; loop < iterate; loop++){  
	RMSE rmse_calc;
	UnscentedKalmanFilter ukf(1, 1, x0, p0);
	ukf.SetProcessNoise(ProcessNoiseCov);
	ukf.SetObservationNoise(ObsNoiseCov);

	for(k = 0; k < T; k+= 1.0){
	  // ==============================
	  // Generate Actual Value
	  // =============================
	  double input = 0.0;
	  processNoise.at<double>(0, 0) = processNoiseGen(engine);
	  process(state, last_state, input);
	  state = state + processNoise;

	  // ==============================
	  // Generate Observation Value
	  // ==============================
	  measurementNoise.at<double>(0, 0) = obsNoiseGen(engine);
	  observation(measurement, state);
	  first_sensor = measurement + measurementNoise;

	  ukf.Update(process, observation, first_sensor);
	  cv::Mat ukf_est = ukf.GetEstimation();

	  outputresult << state.at<double>(0, 0) << " "        // [1] true state
				   << first_sensor.at<double>(0, 0) << " " // [2] first sensor
				   << ukf_est.at<double>(0,0)              // [3] predicted state by UKF
				   << endl;                                
	  
	  rmse_calc.storeData(state.at<double>(0, 0), ukf_est.at<double>(0, 0));
	
	  last_state = state;
	}
	rmse_calc.calculationRMSE();
	cout << "RMSE :" << rmse_calc.getRMSE() << endl;
	rmse_average += rmse_calc.getRMSE();
  }
  rmse_average = rmse_average / (double)iterate;
  cout << "+----------------------------------------+" << endl;
  cout << " Ave RMSE : " << rmse_average << endl;
  cout << "+----------------------------------------+" << endl;
  return 0;
}
