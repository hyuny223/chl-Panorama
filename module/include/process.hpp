#pragma once
#ifndef PROCESS_HPP
#define PROCESS_HPP

void warpImage(const cv::Mat &img,
               cv::Mat &canvas,
               cv::Mat &mask,
               const Eigen::Matrix<double, 3, 3> matrix,
               const Eigen::Matrix<int, 1, 3> offset);
cv::Vec3d bilinear(const cv::Mat &color_pic, double y, double x);
cv::Mat cylinderWarp(const cv::Mat &ori_pic, double focal);
cv::Mat cropCylinder(const cv::Mat &cyl_pic);
void convert2RGB(cv::Mat &fpic, cv::Mat &pic);
void warpCrop(cv::Mat &pic, double focal);
// void warp(const cv::Mat &image,
//           cv::Mat &canvas,
//           const Eigen::Matrix<double, 3, 3> &matrix,
//           const int overlap);
// std::tuple<cv::Mat, int, int> crop(cv::Mat &cyl_pic);

#endif
