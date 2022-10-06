#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include <opencv2/opencv.hpp>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

#include "lm.hpp"

LevenbergMarquardt::LevenbergMarquardt(Eigen::MatrixXd &H)
    : H_(H), initial_H_(H)
{
    /*
        assuming that :
        [1  0  0]   [r11  r12  tx]     [h1  h2  h3]
        [0  1  0] * [r21  r22  ty] = s [h4  h5  h6]
        [0  0 -1]   [r31  r32  tz]     [h7  h8  h9]
    */
    epsilon_1_ = 1e-6;
    epsilon_2_ = 1e-6;
    is_out_ = true;

    max_iter_ = 100;
    lambda_ = 0.001;

    h1 = H_(0, 0), h2 = H_(0, 1), h3 = H_(0, 2);
    h4 = H_(1, 0), h5 = H_(1, 1), h6 = H_(1, 2);
    h7 = H_(2, 0), h8 = H_(2, 1), h9 = H_(2, 2);
}

void LevenbergMarquardt::setParameters(double epsilon_1, double epsilon_2, int max_iter, bool is_out,  double ransac_error)
{
    epsilon_1_ = epsilon_1;
    epsilon_2_ = epsilon_2;
    max_iter_ = max_iter;
    is_out_ = is_out;
    ransac_error_ = ransac_error;
}

void LevenbergMarquardt::addObservation(const double &x, const double &y)
{
    tmp_obs_.push_back(Eigen::Vector2d(x, y));
}

void LevenbergMarquardt::addPoints(const double &x, const double &y)
{
    tmp_pts_.push_back(Eigen::Vector2d(x, y));
}

double LevenbergMarquardt::computeError()
{
    /* 27번 식
    input : obs(8,1), pts(8,1), H(3,3)
    output : error && pose param
    */
    double e{0.0};
    Eigen::Matrix<double, 8, 1> tmp;
    tmp << (h1 * x1 + h2 * y1 + h3) / (h7 * x1 + h8 * y1 + h9),
        (h4 * x1 + h5 * y1 + h6) / (h7 * x1 + h8 * y1 + h9),
        (h1 * x2 + h2 * y2 + h3) / (h7 * x2 + h8 * y2 + h9),
        (h4 * x2 + h5 * y2 + h6) / (h7 * x2 + h8 * y2 + h9),
        (h1 * x3 + h2 * y3 + h3) / (h7 * x3 + h8 * y3 + h9),
        (h4 * x3 + h5 * y3 + h6) / (h7 * x3 + h8 * y3 + h9),
        (h1 * x4 + h2 * y4 + h3) / (h7 * x4 + h8 * y4 + h9),
        (h4 * x4 + h5 * y4 + h6) / (h7 * x4 + h8 * y4 + h9);
    error_ = obs_ - tmp;

    std::cout << "obs_ : " << obs_.transpose() << std::endl;
    std::cout << "prj_ : " << tmp.transpose() << std::endl;

    for (int i = 0; i < 4; ++i)
    {
        e += std::pow(error_(2 * i, 0), 2) + std::pow(error_(2 * i + 1, 0), 2);
    }

    return e;
}

void LevenbergMarquardt::init()
{
    for (int i = 0; i < tmp_obs_.size(); ++i)
    {
        Eigen::Matrix<double, 2, 1> o, p;
        o << tmp_obs_[i][0], tmp_obs_[i][1];
        p << tmp_pts_[i][0], tmp_pts_[i][1];

        obs_.block<2, 1>(2 * i, 0) = o;
        pts_.block<2, 1>(2 * i, 0) = p;
    }
    x1 = pts_(0, 0), x2 = pts_(2, 0), x3 = pts_(4, 0), x4 = pts_(6, 0);
    y1 = pts_(1, 0), y2 = pts_(3, 0), y3 = pts_(5, 0), y4 = pts_(7, 0);

    double h147_sqrt = std::sqrt(h1 * h1 + h4 * h4 + h7 * h7);
    double h258_sqrt = std::sqrt(h2 * h2 + h5 * h5 + h8 * h8);
    scale_ = 2 / (h147_sqrt + h258_sqrt); // 12

    t_ << scale_ * h3, scale_ * h6, -scale_; // 13

    // 14
    double r11 = h1 / h147_sqrt;
    double r21 = h4 / h147_sqrt;
    double r31 = -h7 / h147_sqrt;

    // 15
    double common = r11 * h2 + r21 * h5 - r31 * h8;
    double r12 = h2 - r11 * common;
    double r22 = h5 - r21 * common;
    double r32 = -h8 - r31 * common;

    double l2 = std::sqrt(r12 * r12 + r22 * r22 + r32 * r32);

    r12 /= l2;
    r22 /= l2;
    r32 /= l2;

    // 16
    double r13 = r21 * r32 - r31 * r22;
    double r23 = r31 * r12 - r11 * r32;
    double r33 = r11 * r22 - r21 * r12;

    R_ << r11, r12, r13,
        r21, r22, r23,
        r31, r32, r33;

    // 37
    theta_x_ = std::asin(r32);
    theta_y_ = std::atan2(-r31, r33);
    theta_z_ = std::atan2(-r12, r22);

    sin_x_ = std::sin(theta_x_);
    cos_x_ = std::cos(theta_x_);

    sin_y_ = std::sin(theta_y_);
    cos_y_ = std::cos(theta_y_);

    sin_z_ = std::sin(theta_z_);
    cos_z_ = std::cos(theta_z_);

    param_ = {theta_x_, theta_y_, theta_z_, t_(0, 0), t_(1, 0), t_(2, 0)}; // 19

    // Eigen::Matrix<double, 6, 1> se3;
    // se3 << param_[3], param_[4], param_[5], param_[0], param_[1], param_[2];
    // Sophus::SE3d SE3_ = Sophus::SE3d::exp(se3);

    std::cout << std::endl;
}

void LevenbergMarquardt::updateH()
{
    // 교육 자료에는 안 나와 있는 부분
    theta_x_ = param_[0];
    theta_y_ = param_[1];
    theta_z_ = param_[2];

    sin_x_ = std::sin(theta_x_);
    cos_x_ = std::cos(theta_x_);

    sin_y_ = std::sin(theta_y_);
    cos_y_ = std::cos(theta_y_);

    sin_z_ = std::sin(theta_z_);
    cos_z_ = std::cos(theta_z_);

    Eigen::Matrix<double, 6, 1> se3;
    se3 << param_[3], param_[4], param_[5], param_[0], param_[1], param_[2];
    // Sophus::SE3d SE3_ = Sophus::SE3d::exp(se3);

    Eigen::Matrix<double, 3, 3> tmp_H;
    // tmp_H.block<3, 2>(0, 0) = SE3_.matrix().block<3, 2>(0, 0);
    // tmp_H.block<3, 1>(0, 2) = SE3_.matrix().block<3, 1>(0, 3);
    // lie algebra는 왜 적용이 안 될까??
    tmp_H << cos_y_ * cos_z_ - sin_x_ * sin_y_ * sin_z_, -cos_x_ * sin_z_, param_[3],
        cos_y_ * sin_z_ + sin_x_ * sin_y_ * cos_z_, cos_x_ * cos_z_, param_[4],
        cos_x_ * sin_y_, -sin_x_, -param_[5];
    H_ = tmp_H / tmp_H(2, 2); // tmp_H(2,2)로 나눠야 하나..?

    h1 = H_(0, 0), h2 = H_(0, 1), h3 = H_(0, 2);
    h4 = H_(1, 0), h5 = H_(1, 1), h6 = H_(1, 2);
    h7 = H_(2, 0), h8 = H_(2, 1), h9 = H_(2, 2);
}

// 4개의 점을 기준으로 작성하기
double LevenbergMarquardt::solve()
{
    // first step : compute error
    double e = computeError();

    // second step : homography jacobian -> (26) ~ (34)
    double jg_11 = -cos_x_ * sin_y_ * sin_z_;
    double jg_12 = -sin_y_ * cos_z_ - sin_x_ * cos_y_ * sin_z_;
    double jg_13 = -cos_y_ * sin_z_ - sin_x_ * sin_y_ * cos_z_;
    double jg_14 = 0.0;
    double jg_15 = 0.0;
    double jg_16 = 0.0;

    double jg_21 = sin_x_ * sin_z_;
    double jg_22 = 0.0;
    double jg_23 = -cos_x_ * cos_z_;
    double jg_24 = 0.0;
    double jg_25 = 0.0;
    double jg_26 = 0.0;

    double jg_31 = 0.0;
    double jg_32 = 0.0;
    double jg_33 = 0.0;
    double jg_34 = 1.0;
    double jg_35 = 0.0;
    double jg_36 = 0.0;

    double jg_41 = cos_x_ * sin_y_ * cos_z_;
    double jg_42 = -sin_y_ * sin_z_ + sin_x_ * cos_y_ * cos_z_;
    double jg_43 = cos_y_ * cos_z_ - sin_x_ * sin_y_ * sin_z_;
    double jg_44 = 0.0;
    double jg_45 = 0.0;
    double jg_46 = 0.0;

    double jg_51 = -sin_x_ * cos_z_;
    double jg_52 = 0.0;
    double jg_53 = -cos_x_ * sin_z_;
    double jg_54 = 0.0;
    double jg_55 = 0.0;
    double jg_56 = 0.0;

    double jg_61 = 0.0;
    double jg_62 = 0.0;
    double jg_63 = 0.0;
    double jg_64 = 0.0;
    double jg_65 = 1.0;
    double jg_66 = 0.0;

    double jg_71 = -sin_x_ * sin_y_;
    double jg_72 = cos_x_ * cos_y_;
    double jg_73 = 0.0;
    double jg_74 = 0.0;
    double jg_75 = 0.0;
    double jg_76 = 0.0;

    double jg_81 = -cos_x_;
    double jg_82 = 0.0;
    double jg_83 = 0.0;
    double jg_84 = 0.0;
    double jg_85 = 0.0;
    double jg_86 = 0.0;

    double jg_91 = 0.0;
    double jg_92 = 0.0;
    double jg_93 = 0.0;
    double jg_94 = 0.0;
    double jg_95 = 0.0;
    double jg_96 = -1.0;

    Jg_ << jg_11, jg_12, jg_13, jg_14, jg_15, jg_16,
        jg_21, jg_22, jg_23, jg_24, jg_25, jg_26,
        jg_31, jg_32, jg_33, jg_34, jg_35, jg_36,
        jg_41, jg_42, jg_43, jg_44, jg_45, jg_46,
        jg_51, jg_52, jg_53, jg_54, jg_55, jg_56,
        jg_61, jg_62, jg_63, jg_64, jg_65, jg_66,
        jg_71, jg_72, jg_73, jg_74, jg_75, jg_76,
        jg_81, jg_82, jg_83, jg_84, jg_85, jg_86,
        jg_91, jg_92, jg_93, jg_94, jg_95, jg_96;

    // third step : projection jacobian -> (24) ~ (25)
    for (size_t i = 0; i < tmp_pts_.size(); i++)
    {
        Eigen::MatrixXd jf_block1(1, 9), jf_block2(1, 9);

        const Eigen::Vector2d &pt = tmp_pts_.at(i);
        const double &x = pt(0);
        const double &y = pt(1);

        double j1_row_2 = h1 * x + h2 * y + h3;                    // x
        double j2_row_2 = h4 * x + h5 * y + h6;                    // y
        double j_row_12_common = h7 * x + h8 * y + h9;             // z
        double j_row_3_common = j_row_12_common * j_row_12_common; // z*z

        double jf_11 = x / j_row_12_common;
        double jf_12 = y / j_row_12_common;
        double jf_13 = 1.0 / j_row_12_common;
        double jf_14 = 0.0;
        double jf_15 = 0.0;
        double jf_16 = 0.0;
        double jf_17 = -(j1_row_2 / j_row_3_common) * x;
        double jf_18 = -(j1_row_2 / j_row_3_common) * y;
        double jf_19 = -(j1_row_2 / j_row_3_common) * 1.0;
        jf_block1 << jf_11, jf_12, jf_13, jf_14, jf_15, jf_16, jf_17, jf_18, jf_19;

        double jf_21 = 0.0;
        double jf_22 = 0.0;
        double jf_23 = 0.0;
        double jf_24 = x / j_row_12_common;
        double jf_25 = y / j_row_12_common;
        double jf_26 = 1.0 / j_row_12_common;
        double jf_27 = -(j2_row_2 / j_row_3_common) * x;
        double jf_28 = -(j2_row_2 / j_row_3_common) * y;
        double jf_29 = -(j2_row_2 / j_row_3_common) * 1.0;
        jf_block2 << jf_21, jf_22, jf_23, jf_24, jf_25, jf_26, jf_27, jf_28, jf_29;

        Jf_.block<1, 9>(2 * i, 0) = jf_block1;
        Jf_.block<1, 9>(2 * i + 1, 0) = jf_block2;
    }
    J_ = Jf_ * Jg_; // (8 x 9), (9 x 6) = (8 x 6)
    Eigen::Matrix<double, 6, 8> Jt = J_.transpose();
    He_ = Jt * J_;
    Eigen::Matrix<double, 6, 6> diag = He_.diagonal().asDiagonal();

    param_ += (He_ + lambda_ * diag).inverse() * Jt * error_; // p = p + (Jt*J + lambda *diag(Jt*J)).inv() * Jt(b - f(g(p)))
    std::cout << "updated param_ : " << param_.transpose() << std::endl;

    return e;
}

void LevenbergMarquardt::run()
{
    init();
    double last_cost = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        double cost = solve(); // 22

        lambda_ = cost < last_cost ? lambda_ * 0.8 : lambda_ * 1.2;

        last_cost = cost;
        std::cout << "iter : " << iter << " --> "
                  << "cost : " << last_cost << std::endl;
        updateH();
        std::cout << "-----------" << std::endl;
        std::cout << std::endl;
    }
    last_cost_ = last_cost;
}

Eigen::Matrix<double, 3, 3> LevenbergMarquardt::model()
{
    return H_;
    // return last_cost_ < ransac_error_ ? H_ : initial_H_;
}
