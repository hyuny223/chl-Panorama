#pragma once

#ifndef HOMOGRAPHY_HPP
#define HOMOGRAPHY_HPP

#include <iostream>
#include <vector>
#include <random>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "RANSAC.hpp"

class Homography
{
protected:
    Eigen::MatrixXd B_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd source_, target_, proj_;
    std::vector<cv::Point2d> source_pts_, target_pts_;
    std::vector<int> indices_;
    std::vector<int> inliers_;

    double delta_;
    int n_, pts_n_;

public:
    double error_;
    int inliers_cnt_{0};

public:
    Homography() = default;
    Homography(const std::vector<cv::Point2d> &source_pts,
               const std::vector<cv::Point2d> &target_pts);
    bool check(int n);
    void sampling(std::mt19937 &gen);
    void run(double delta, std::mt19937 &gen);
    void computeSVD();
    void computeB();
    // void computeC(const Eigen::BDCSVD<Eigen::MatrixXd> &svd);
    void computeC(const Eigen::JacobiSVD<Eigen::MatrixXd> &svd);
    void compute2D();
    void computeInliers();
    std::vector<int> getIndices();
    Eigen::MatrixXd &model();
};

#endif
