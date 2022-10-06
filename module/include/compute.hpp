#pragma once

#ifndef DETECT_HPP
#define DETECT_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

auto detect(const cv::Mat &image1, const cv::Mat &image2)
{
    std::vector<cv::KeyPoint> keys1, keys2;
    cv::Mat desc1, desc2;
    cv::Mat cat = cv::Mat::ones(cv::Size(image1.cols / 2, image1.rows), CV_8UC1) * 255;

    cv::Mat mask1, mask2;

    // mask1 = cv::Mat::zeros(cv::Size(image1.cols / 2, image1.rows), CV_8UC1);
    // mask2 = cv::Mat::zeros(cv::Size(image1.cols / 2, image1.rows), CV_8UC1);

    // cv::hconcat(mask1, cat, mask1);
    // cv::hconcat(cat, mask2, mask2);

    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create(2000);

    detector->detectAndCompute(image1, mask1, keys1, desc1);
    detector->detectAndCompute(image2, mask2, keys2, desc2);

    return std::tuple{keys1, desc1, keys2, desc2};
}

auto matching(const cv::Mat &img1,
              const auto &keys1,
              const auto &desc1,
              const cv::Mat &img2,
              const auto &keys2,
              const auto &desc2)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);

    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good;
    std::vector<cv::Point2d> good1, good2;
    matcher->knnMatch(desc1, desc2, matches, 2);

    for (auto &m : matches)
    {
        if (m.at(0).distance / m.at(1).distance < 0.5)
        {
            good.emplace_back(m.at(0));
        }
    }

    good1.reserve(good.size());
    good2.reserve(good.size());

    for (auto &g : good)
    {
        good1.push_back(keys1[g.queryIdx].pt);
        good2.push_back(keys2[g.trainIdx].pt);
    }

    return std::tuple{good1, good2};
}

#endif
