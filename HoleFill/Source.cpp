#include "holeFilling.cuh"

int main() {
	warmupGPU();
	cv::Mat image;
	//image = cv::imread(R"(path/to/image)");
	image = cv::imread(R"(path/to/image)");
	cv::Mat result(image.rows, image.cols, CV_8U);

	if (!image.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}

	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::threshold(image, image, 125, 255, 0);
	displayImage(image, "hole image", false);


	double avg_time = 0.0;
	const int iter = 50;
	HoleFilling hole_fill;
	for (int i = 0; i < iter; i++) {
		auto start = std::chrono::system_clock::now();
		hole_fill.Process(image, result);
		auto end = std::chrono::system_clock::now();

		// time
		std::chrono::duration<double> elapsed_seconds = end - start;
		
		avg_time += elapsed_seconds.count();
	}
	avg_time /= iter;
	std::cout << "time consume: " << avg_time * 1000.0 << " ms" << std::endl;

	displayImage(result, "result", false);
	//std::vector<int> compression_params;
	//compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(9);
	//cv::imwrite("result.png", result, compression_params);
	return 0;
}


