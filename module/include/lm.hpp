#pragma once

#ifndef LM_HPP
#define LM_HPP

#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>

#include <opencv2/opencv.hpp>

class LevenbergMarquardt
{

private:
    Eigen::Matrix<double, 8, 9> Jf_;
    Eigen::Matrix<double, 9, 6> Jg_;
    Eigen::Matrix<double, 8, 6> J_;  // Jf_ * Jg_
    Eigen::Matrix<double, 6, 6> He_; // hessian(J.t * J)
    Eigen::Matrix<double, 3, 3> H_;  // homography
    Eigen::Matrix<double, 3, 3> initial_H_;

    std::vector<Eigen::Vector2d> tmp_obs_, tmp_pts_; // gt, want to adjust

    Eigen::Matrix<double, 3, 1> t_;
    Eigen::Matrix<double, 3, 3> R_;

    Eigen::Matrix<double, 6, 1> param_;

    Eigen::Matrix<double, 8, 1> obs_, pts_, error_;

    double lambda_;
    double scale_;
    double ransac_error_;
    double last_cost_;

    /* parameters */
    double epsilon_1_, epsilon_2_;
    int max_iter_;
    bool is_out_;

    double x1, x2, x3, x4;
    double y1, y2, y3, y4;

    double h1, h2, h3;
    double h4, h5, h6;
    double h7, h8, h9;

    double theta_x_, theta_y_, theta_z_;

    double sin_x_, sin_y_, sin_z_;
    double cos_x_, cos_y_, cos_z_;

public:
    LevenbergMarquardt(Eigen::MatrixXd &H);
    void setParameters(double epsilon_1, double epsilon_2, int max_iter, bool is_out, double ransac_error);
    void addObservation(const double &x, const double &y); // 이 부분은 vector<cv::Point>로 바로 넣어도 되지만, 훗날 동적 포인터만 쓸 날을 위하여 이렇게 작성함
    void addPoints(const double &x, const double &y);
    double computeError();
    void updateH();
    void init();
    double solve();
    void run();
    Eigen::Matrix<double, 3, 3> model();

}; // class LevenbergMarquardt

#endif
