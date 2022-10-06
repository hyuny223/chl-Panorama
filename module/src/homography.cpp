#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "homography.hpp"

Homography::Homography(const std::vector<cv::Point2d> &source_pts,
                       const std::vector<cv::Point2d> &target_pts)
{
    source_pts_ = source_pts;
    target_pts_ = target_pts;

    pts_n_ = source_pts.size();

    Eigen::MatrixXd tmp_s(3, pts_n_), tmp_t(3, pts_n_);
    for (int i = 0; i <pts_n_; ++i)
    {
        Eigen::MatrixXd s(3, 1), t(3, 1);
        s << source_pts_[i].x, source_pts_[i].y, 1.0;
        t << target_pts_[i].x, target_pts_[i].y, 1.0;

        tmp_s.block<3, 1>(0, i) = s;
        tmp_t.block<3, 1>(0, i) = t;
    }

    source_ = tmp_s;
    target_ = tmp_t;
}

bool Homography::check(int n)
{
    for (auto i : indices_)
    {
        if (i == n)
        {
            return false;
        }
    }
    return true;
}

void Homography::sampling(std::mt19937 &gen)
{
    std::uniform_int_distribution<int> dis(0, pts_n_ - 1);
    indices_.clear();
    indices_.reserve(4);

    int cnt = 0;
    while (cnt < 4)
    {
        int idx = dis(gen);
        if (check(idx))
        {
            indices_.push_back(idx); // 중복 검사 안 해줘도 될까? --> 해주어야 함
            ++cnt;
        }
    }
}

void Homography::run(double delta, std::mt19937 &gen)
{
    delta_ = delta;
    sampling(gen);
    computeSVD();
    computeInliers();
}

void Homography::computeSVD()
{
    computeB();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::BDCSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // B_.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(C_);
    computeC(svd);
    compute2D();
}

void Homography::computeB()
{
    Eigen::MatrixXd B(8, 9);
    // cv::Mat board(1000, 1000, CV_8UC3, cv::Scalar::all(255));

    int row = 1;
    for (const int i : indices_)
    {
        cv::Point2d p = source_pts_[i];
        cv::Point2d q = target_pts_[i];
        // cv::circle(board, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 3, cv::Scalar(255, 0, 0), 3);
        // cv::circle(board, cv::Point(static_cast<int>(q.x), static_cast<int>(q.y)), 3, cv::Scalar(0, 0, 255), 3);

        double x = p.x, y = p.y;
        double u = q.x, v = q.y;

        Eigen::MatrixXd A1(1, 9), A2(1, 9);
        A1 << x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, -u;
        A2 << 0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, -v;

        B.block<1, 9>(2 * row - 2, 0) = A1;
        B.block<1, 9>(2 * row - 1, 0) = A2;
        ++row;
    }
    // cv::imshow("a", board);
    // cv::waitKey(0);
    B_ = B;
}

void Homography::computeC(const Eigen::JacobiSVD<Eigen::MatrixXd> &svd)
{
    Eigen::Matrix<double, 3, 3> C;
    auto tmp = svd.matrixV().col(8);

    C.block<1, 3>(0, 0) = tmp.block<3, 1>(0, 0);
    C.block<1, 3>(1, 0) = tmp.block<3, 1>(3, 0);
    C.block<1, 3>(2, 0) = tmp.block<3, 1>(6, 0);

    C_ = C;
}

void Homography::compute2D()
{
    auto T = (C_ * source_);                            // col이 3인 형태 (x, y, a)
    auto proj = T.array().rowwise() / T.row(2).array(); // (x/a, y/a, 1)

    proj_ = proj;
}

void Homography::computeInliers()
{
    int cnt{0};
    double error{0};
    inliers_.clear();

    for (int i = 0; i < pts_n_; ++i)
    {

        auto p_x = proj_(0, i);
        auto p_y = proj_(1, i);
        auto n_x = target_(0, i);
        auto n_y = target_(1, i);

        // std::cout << "proj x : " << p_x << ", proj y : " << p_y << std::endl;
        // std::cout << "target x : " << n_x << ", target y : " << n_y << std::endl;

        double e = std::sqrt((n_x - p_x) * (n_x - p_x) + (n_y - p_y) * (n_y - p_y));
        if(e < delta_)
        {
            ++cnt;
            inliers_.emplace_back(i);
        }
        error += e;
    }
    // std::cout << "--------" << std::endl;
    inliers_cnt_ = cnt;
    error_ = error / pts_n_;
}

std::vector<int> Homography::getIndices()
{
    return indices_;
}

Eigen::MatrixXd &Homography::model()
{
    return C_;
}
