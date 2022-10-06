#pragma once
#ifndef PROSAC_HPP
#define PROSAC_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "opencv2/opencv.hpp"

#include "homography.hpp"

template <typename MODEL>
class RANSAC
{
protected:
    MODEL model_, candidate_;

    std::vector<int> candidate_model_inliers_;
    double candidate_model_inliers_cnt_ = std::numeric_limits<double>::lowest();
    double candidate_model_error_ = std::numeric_limits<double>::infinity();

    int iteration_, pts_size_, m_;
    double p_, eps_, delta_, th_;
    double minimum_inliers_cnt_;

public:
    RANSAC(MODEL &model,
           int pts_size,
           double p = 0.99,
           double eps = 0.6,
           double delta = 3.0,
           int m = 4)
        : model_(model), pts_size_(pts_size), p_(p), eps_(eps), delta_(delta), m_(m)
    {
        iteration_ = computeN();
        minimum_inliers_cnt_ = (1 - eps) * pts_size;
        std::cout << "iter : " << iteration_ << std::endl;
        std::cout << "min : " << minimum_inliers_cnt_ << std::endl;
        th_ = 1.0;
    };

    MODEL &run()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int iter = 0; iter < iteration_; ++iter)
        {
            if (iterate(gen))
            {
                if (candidate_model_error_ < th_)
                {
                    std::cout << "[ MODEL COMPLETED ]" << std::endl;
                    std::cout << "the number of inliers of model : " << candidate_.inliers_cnt_ << std::endl;
                    std::cout << "the lowest error of model : " << candidate_.error_ << std::endl;
                    std::cout << "homography : \n"
                              << candidate_.model() << std::endl;
                    return candidate_;
                }
            }
        }
        std::cout << "[ CANDIDATE COMPLETED ]" << std::endl;
        std::cout << "the number of inliers so far : " << candidate_.inliers_cnt_ << std::endl;
        std::cout << "the lowest error so far : " << candidate_.error_ << std::endl;
        std::cout << "homography : \n"
                  << candidate_.model() << std::endl;

        return candidate_;
    }
    bool iterate(std::mt19937 &gen)
    {

        model_.run(delta_, gen);
        // std::cout << "[ CANDIDATE ]" << std::endl;
        // std::cout << "the number of inliers so far : " << model_.inliers_cnt_ << std::endl;
        // std::cout << "the lowest error so far : " << model_.error_ << std::endl;

        if (model_.inliers_cnt_ > minimum_inliers_cnt_ &&
            model_.inliers_cnt_ > candidate_model_inliers_cnt_ &&
            // 0.01 < model_.error_ &&
            model_.error_ < candidate_model_error_)
        {
            candidate_model_error_ = model_.error_;
            candidate_model_inliers_cnt_ = model_.inliers_cnt_;
            candidate_ = model_;
            return true;
        }
        return false;
    }

    int computeN()
    {
        return std::round(std::log10(1 - p_) / std::log10(1 - std::pow(1 - eps_, m_)));
    }
};

#endif
