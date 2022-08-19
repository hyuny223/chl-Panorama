#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"

void processImages(const cv::Mat& leftImage, cv::Mat& rightImage);
void cropBlackArea(cv::Mat& image, const std::string& direction);
void warpCylinder(cv::Mat& image);
void cropBlackArea11(cv::Mat& image, const std::string& direction);
