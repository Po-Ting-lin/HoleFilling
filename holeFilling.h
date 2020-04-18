#pragma once
#include<opencv2/opencv.hpp>

#define CV_8U_DYNAMICRANGE 256

class HoleFilling
{
private:
	cv::Mat BorderImageInit(cv::Mat& src);
	cv::Mat SpeedBorderImageInit(cv::Mat& src);
public:
	void process(cv::Mat& src, cv::Mat& dst);
};

static void displayImage(const cv::Mat& image, const char* name, bool mag) {
	cv::Mat Out;
	if (mag) {
		cv::resize(image, Out, cv::Size(), 5, 5);
	}
	else {
		image.copyTo(Out);
	}
	namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::imshow(name, Out);
	cv::waitKey(0);
}
