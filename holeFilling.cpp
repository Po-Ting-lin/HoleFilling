#include "holeFilling.h"


void HoleFilling::process(cv::Mat& src, cv::Mat& dst) {
    cv::Mat image, preMarker, cImage, curImage, nextImage;
    int imageSize = src.rows * src.cols;

    cvtColor(src, src, cv::COLOR_BGR2GRAY);

    cv::threshold(src, image, 125, CV_8U_DYNAMICRANGE - 1, 0);
    //copyMakeBorder(src, image, 1, 1, 1, 1, CV_8U, 0);
    cv::bitwise_not(image, cImage);

    // init marker
    image.copyTo(preMarker);
    uchar* preMarkerPtr = preMarker.data;
    bool* visitedArr = new bool[imageSize] {};
    bool* curvisited = visitedArr;

    // left to right
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {

            // black and not visited
            if (!(*preMarkerPtr) && !(*curvisited)) {
                *preMarkerPtr = CV_8U_DYNAMICRANGE - 1;
                *curvisited = true;
                preMarkerPtr++;
                curvisited++;
            }
            else {
                preMarkerPtr = preMarker.data + r * src.cols;
                curvisited = visitedArr + r * src.cols;
                break;
            }
        }
    }

    // right to left
    for (int r = 0; r < src.rows; r++) {
        preMarkerPtr = preMarker.data + (r + 1) * src.cols - 1;
        curvisited = visitedArr + (r + 1) * src.cols - 1;
        for (int c = 0; c < src.cols; c++) {

            // black and not visited
            if (!(*preMarkerPtr) && !(*curvisited)) {
                *preMarkerPtr = CV_8U_DYNAMICRANGE - 1;
                *curvisited = true;
                preMarkerPtr--;
                curvisited--;
            }
            else {
                break;
            }
        }
    }
    preMarkerPtr = preMarker.data;
    curvisited = visitedArr;
    for (int r = 0; r < imageSize; r++, preMarkerPtr++, curvisited++) {
        if (!(*curvisited)) {
            *preMarkerPtr = 0;
        }
    }

    delete[] visitedArr;
    visitedArr = nullptr;
    preMarker.copyTo(curImage);
    uchar* dilationKernel = new uchar[9]{ 0,1,0,1,1,1,0,1,0 };
    cv::Mat element(3, 3, CV_8U, &dilationKernel);

    //displayImage(curImage, "curImage", false);
    //displayImage(cImage, "cImage", false);

    int count = 0;
    while (1) {
        cv::dilate(curImage, nextImage, element, cv::Point(-1, -1), 1);
        cv::bitwise_and(nextImage, cImage, nextImage);
        //displayImage(nextImage, "nextImage", false);
        if (std::equal(curImage.begin<uchar>(), curImage.end<uchar>(), nextImage.begin<uchar>())) {
            break;
        }
        else {
            nextImage.copyTo(curImage);
            count++;
        }
        if (count > 5000) {
            break;
        }
    }
    cv::bitwise_not(nextImage, nextImage);
    cv::bitwise_and(nextImage, cImage, nextImage);
    cv::bitwise_or(nextImage, image, nextImage);
    nextImage.copyTo(dst);
}
