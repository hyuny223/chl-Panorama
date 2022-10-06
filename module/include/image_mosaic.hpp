#pragma once

#ifndef IMAGE_MOSAIC_HPP
#define IMAGE_MOSIAC_HPP

#include <iostream>
#include <vector>
#include <unordered_map>

#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"

class ImageMosaic
{
private:
    int mid_, images_num_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> matrices_;
    std::vector<int> overlap_;
    std::vector<int> images_rows_, images_cols_;
    std::vector<cv::Mat> images_;

public:
    ImageMosaic(const std::vector<std::string> &names);
    void mosaic();
    void stitch(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices);
    void sortH(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices);
    void generateCanvas(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices,
                        cv::Mat &canvas,
                        cv::Mat &mask,
                        Eigen::Matrix<int, 1, 3> &offset);
    void computeMove(Eigen::Matrix<double, 3, 3> &matrix,
                     const int row,
                     const int col,
                     Eigen::Matrix<int, 1, 3> &min_crd,
                     Eigen::Matrix<int, 1, 3> &max_crd);
};

#endif
