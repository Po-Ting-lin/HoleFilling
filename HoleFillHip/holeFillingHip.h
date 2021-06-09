#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <math.h>
#include "hip/hip_runtime.h"
#include "hipCheck.h"

#define TIMER true
#define BLOCK_DIM 32
#define CV_8U_DYNAMICRANGE 256

class HoleFilling
{
private:
    int maxIters;

    void borderImageInit(cv::Mat& src, cv::Mat& dst);
    void speedBorderImageInit(cv::Mat& src, cv::Mat& dst);
    void fillprocess(cv::Mat& src, cv::Mat& premarker, cv::Mat& dst);

public:
    HoleFilling() {
        maxIters = 5000;
    };

    void Process(cv::Mat& src, cv::Mat& dst);
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

void warmupGPU();

static inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void NotKernel(bool* src, bool* dst, int size);
__global__ void NaiveDilationAndKernel(bool* src, bool* dst, bool* ref, int width, int height);
__global__ void NaiveDilationAndSmemKernel(bool* src, bool* dst, bool* ref, int width, int height, int tileSize);
__global__ void NotAndOrKernel(bool* src, bool* andRef, bool* orRef, int size);
__global__ void IsEqualReductionKernel(bool* src, bool* ref, bool* isEqualReduced, int size);
__global__ void IsEqualReductionKernel(bool* isEqual, bool* isEqualReduced, int size);
__device__ void reduction(bool* smem, bool* dst);