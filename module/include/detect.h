#pragma once
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "process.h"


class Detect
{
    private:
        cv::Mat mSrcImage, mTargetImage, mHomoGraphy;
        std::vector<cv::KeyPoint> mvSrcKeypoint, mvTargetKeypoint;

        cv::Mat mSrcDesc, mTargetDesc;

        std::vector<cv::Point2d> mvSrcGood, mvTargetGood;

    public:
        Detect(cv::Mat& src, cv::Mat rarget);

        void detectFeatures();
        void matchFeatures();
        void findMatchingMatrix();
        void transformImage(const std::string& direction);
        cv::Mat getSrcImage();

};
