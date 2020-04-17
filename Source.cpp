#include<iostream>
#include<opencv2/opencv.hpp>
#include"holeFilling.h"

using namespace cv;
using namespace std;

int main() {
	Mat image, result;
	image = imread(R"(C:\Users\brian.AMO\Desktop\hole.png)");

	if (!image.data) {
		cout << "Error: the image wasn't correctly loaded." << endl;
		return -1;
	}
	
	displayImage(image, "hole image", false);

	HoleFilling obj;	
	///*******************************************************/
	auto start = std::chrono::system_clock::now();
	obj.process(image, result);
	auto end = std::chrono::system_clock::now();

	// time
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << "time consume: " << elapsed_seconds.count() << endl;
	/*******************************************************/
	
	displayImage(result, "result", false);

	return 0;
}


