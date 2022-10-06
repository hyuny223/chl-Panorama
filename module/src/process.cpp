#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>

#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "process.hpp"

void warpImage(const cv::Mat &img,
               cv::Mat &canvas,
               cv::Mat &mask,
               const Eigen::Matrix<double, 3, 3> matrix,
               const Eigen::Matrix<int, 1, 3> offset)
{
    double wx[2];
    double wy[2];

    for (int r = 0; r < canvas.rows; r++)
    {
        for (int c = 0; c < canvas.cols; c++)
        {
            if(canvas.ptr<cv::Vec3b>(r)[c] == cv::Vec3b(0, 0, 0))
            {
                cv::Vec3b pixel = 0;
                double ratio{0.0};

                Eigen::MatrixXd point(3, 1);
                point << c, r, 1.0;

                Eigen::MatrixXd proj = matrix * point;
                proj = proj / proj(2, 0);

                double py = proj(1, 0) + offset(0,1);
                double px = proj(0, 0) + offset(0,0);

                wx[1] = px - std::floor(px);
                wx[0] = 1.0 - wx[1];

                wy[1] = py - std::floor(py);
                wy[0] = 1.0 - wy[1];

                int x = static_cast<int>(std::floor(px));
                int y = static_cast<int>(std::floor(py));

                if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
                {
                    pixel += wx[0] * wy[0] * img.ptr<cv::Vec3b>(y)[x];
                    ratio += wx[0] * wy[0];
                }

                if (x + 1 >= 0 && x + 1 < img.cols && y >= 0 && y < img.rows)
                {
                    pixel += wx[1] * wy[0] * img.ptr<cv::Vec3b>(y)[x+1];
                    ratio += wx[1] * wy[0];
                }

                if (x >= 0 && x < img.cols && y + 1 >= 0 && y + 1 < img.rows)
                {
                    pixel += wx[0] * wy[1] * img.ptr<cv::Vec3b>(y+1)[x];
                    ratio += wx[0] * wy[1];
                }

                if (x + 1 >= 0 && x + 1 < img.cols && y + 1 >= 0 && y + 1 < img.rows)
                {
                    pixel += wx[1] * wy[1] * img.ptr<cv::Vec3b>(y+1)[x+1];
                    ratio += wx[1] * wy[1];
                }
                cv::Vec3b val = pixel / ratio;

                // std::cout << "pixel : " << pixel << std::endl;
                // std::cout << "ratio : " << ratio << std::endl;
                // std::cout << "val : " << val << std::endl;
                canvas.ptr<cv::Vec3b>(r)[c] = val;
            }
        }
    }
    cv::imshow("canvas", canvas);
    cv::waitKey(0);
}

cv::Vec3f bicubic(const std::vector<std::vector<cv::Vec3f>> &p, float x, float y)
{
    // 다시 작성해봐야할 듯..
    cv::Vec3f a00 = p[1][1];
    cv::Vec3f a01 = -.5 * p[1][0] + .5 * p[1][2];
    cv::Vec3f a02 = p[1][0] - 2.5 * p[1][1] + 2 * p[1][2] - .5 * p[1][3];
    cv::Vec3f a03 = -.5 * p[1][0] + 1.5 * p[1][1] - 1.5 * p[1][2] + .5 * p[1][3];
    cv::Vec3f a10 = -.5 * p[0][1] + .5 * p[2][1];
    cv::Vec3f a11 = .25 * p[0][0] - .25 * p[0][2] - .25 * p[2][0] + .25 * p[2][2];
    cv::Vec3f a12 = -.5 * p[0][0] + 1.25 * p[0][1] - p[0][2] + .25 * p[0][3] +
                    .5 * p[2][0] - 1.25 * p[2][1] + p[2][2] - .25 * p[2][3];
    cv::Vec3f a13 = .25 * p[0][0] - .75 * p[0][1] + .75 * p[0][2] - .25 * p[0][3] -
                    .25 * p[2][0] + .75 * p[2][1] - .75 * p[2][2] + .25 * p[2][3];
    cv::Vec3f a20 = p[0][1] - 2.5 * p[1][1] + 2 * p[2][1] - .5 * p[3][1];
    cv::Vec3f a21 = -.5 * p[0][0] + .5 * p[0][2] + 1.25 * p[1][0] - 1.25 * p[1][2] -
                    p[2][0] + p[2][2] + .25 * p[3][0] - .25 * p[3][2];
    cv::Vec3f a22 = p[0][0] - 2.5 * p[0][1] + 2 * p[0][2] - .5 * p[0][3] - 2.5 * p[1][0] +
                    6.25 * p[1][1] - 5 * p[1][2] + 1.25 * p[1][3] + 2 * p[2][0] - 5 * p[2][1] +
                    4 * p[2][2] - p[2][3] - .5 * p[3][0] + 1.25 * p[3][1] - p[3][2] + .25 * p[3][3];
    cv::Vec3f a23 = -.5 * p[0][0] + 1.5 * p[0][1] - 1.5 * p[0][2] + .5 * p[0][3] + 1.25 * p[1][0] -
                    3.75 * p[1][1] + 3.75 * p[1][2] - 1.25 * p[1][3] - p[2][0] + 3 * p[2][1] -
                    3 * p[2][2] + p[2][3] + .25 * p[3][0] - .75 * p[3][1] + .75 * p[3][2] - .25 * p[3][3];
    cv::Vec3f a30 = -.5 * p[0][1] + 1.5 * p[1][1] - 1.5 * p[2][1] + .5 * p[3][1];
    cv::Vec3f a31 = .25 * p[0][0] - .25 * p[0][2] - .75 * p[1][0] + .75 * p[1][2] +
                    .75 * p[2][0] - .75 * p[2][2] - .25 * p[3][0] + .25 * p[3][2];
    cv::Vec3f a32 = -.5 * p[0][0] + 1.25 * p[0][1] - p[0][2] + .25 * p[0][3] + 1.5 * p[1][0] -
                    3.75 * p[1][1] + 3 * p[1][2] - .75 * p[1][3] - 1.5 * p[2][0] + 3.75 * p[2][1] -
                    3 * p[2][2] + .75 * p[2][3] + .5 * p[3][0] - 1.25 * p[3][1] + p[3][2] - .25 * p[3][3];
    cv::Vec3f a33 = .25 * p[0][0] - .75 * p[0][1] + .75 * p[0][2] - .25 * p[0][3] - .75 * p[1][0] +
                    2.25 * p[1][1] - 2.25 * p[1][2] + .75 * p[1][3] + .75 * p[2][0] - 2.25 * p[2][1] +
                    2.25 * p[2][2] - .75 * p[2][3] - .25 * p[3][0] + .75 * p[3][1] -
                    .75 * p[3][2] + .25 * p[3][3];
    float x2 = x * x;
    float x3 = x2 * x;
    return a00 + (a01 + (a02 + a03 * y) * y) * y +
           (a10 + (a11 + (a12 + a13 * y) * y) * y) * x +
           (a20 + (a21 + (a22 + a23 * y) * y) * y) * x2 +
           (a30 + (a31 + (a32 + a33 * y) * y) * y) * x3;
};

void warpCrop(cv::Mat &pic, double focal)
{
    cv::Mat cyl = cylinderWarp(pic, focal);
    cv::Mat crop = cropCylinder(cyl);
    convert2RGB(crop, pic);
}

cv::Vec3d bilinear(const cv::Mat &color_pic, double y, double x)
{
    int int_x = static_cast<int>(x);
    int int_y = static_cast<int>(y);

    if (x < 0 || int_x >= color_pic.rows - 1)
    {
        return cv::Vec3d(-1, -1, -1);
    }
    if (y < 0 || int_y >= color_pic.cols - 1)
    {
        return cv::Vec3d(-1, -1, -1);
    }

    cv::Vec3d a = color_pic.ptr<cv::Vec3d>(int_x)[int_y];
    cv::Vec3d b = color_pic.ptr<cv::Vec3d>(int_x + 1)[int_y];
    cv::Vec3d c = color_pic.ptr<cv::Vec3d>(int_x)[int_y + 1];
    cv::Vec3d d = color_pic.ptr<cv::Vec3d>(int_x + 1)[int_y + 1];

    float u = x - int_x;
    float v = y - int_y;

    return ((1 - u) * a + u * b) * (1 - v) + ((1 - u) * c + u * d) * v;
}

cv::Mat cylinderWarp(const cv::Mat &ori_pic, double focal)
{
    cv::Mat pic, cyl_pic;
    ori_pic.convertTo(pic, CV_64FC3);

    double width = pic.cols - 1;
    double height = pic.rows - 1;
    double mx_t = focal * std::atan(width / 2.f / focal) * 2.f;

    cyl_pic = cv::Mat(pic.rows, std::ceil(mx_t), CV_64FC3);
    for (int r = 0; r < cyl_pic.rows; ++r)
    {
        for (int c = 0; c < cyl_pic.cols; ++c)
        {
            double theta = (c - (cyl_pic.cols - 1) / 2.f);
            double hval = (r - (cyl_pic.rows - 1) / 2.f);
            double y = std::tan(theta / focal) * focal;
            double x = hval / focal * std::sqrt(y * y + focal * focal);
            double xshift = x + height / 2.f;
            double yshift = y + width / 2.f;

            cyl_pic.ptr<cv::Vec3d>(r)[c] = bilinear(pic, yshift, xshift);
        }
    }
    return cyl_pic;
}

cv::Mat cropCylinder(const cv::Mat &cyl_pic)
{
    cv::Mat crp_pic;
    int st = -1, end = cyl_pic.rows;
    for (int c = 0; c < cyl_pic.cols; ++c)
    {
        int min_pos = cyl_pic.rows, max_pos = -1;
        for (int r = 0; r < cyl_pic.rows; ++r)
        {
            if (cyl_pic.ptr<cv::Vec3d>(r)[c][0] != -1)
            {
                min_pos = std::min(min_pos, r);
                max_pos = std::max(max_pos, r);
            }
        }
        st = std::max(st, min_pos);
        end = std::min(end, max_pos);
    }

    crp_pic = cv::Mat(end - st + 1, cyl_pic.cols, CV_64FC3);
    for (int c = 0; c < cyl_pic.cols; ++c)
    {
        for (int r = st; r <= end; ++r)
            crp_pic.ptr<cv::Vec3d>(r - st)[c] = cyl_pic.ptr<cv::Vec3d>(r)[c];
    }

    return crp_pic;
}

void convert2RGB(cv::Mat &fpic, cv::Mat &pic)
{
    pic = cv::Mat(fpic.rows, fpic.cols, CV_8UC3);

    for (int y = 0; y < fpic.rows; ++y)
    {
        for (int x = 0; x < fpic.cols; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                double intensity = fpic.ptr<cv::Vec3d>(y)[x][c];
                pic.ptr<cv::Vec3b>(y)[x][c] = static_cast<int>(std::max(intensity, 0.0));
            }
        }
    }
}

// void warp(const cv::Mat &image,
//           cv::Mat &canvas,
//           const Eigen::Matrix<double, 3, 3> &matrix,
//           const int overlap)
// {
//     cv::Vec3b pixel = 0;
//     double ratio = 0.0;

//     double wx[2];
//     double wy[2];

//     for (int r = 0; r < canvas.rows; ++r)
//     {
//         for (int c = 0; c < canvas.cols; ++c)
//         {
//             pixel = 0;
//             ratio = 0.0;
//             Eigen::MatrixXd point(3, 1);
//             point << c, r, 1.0;

//             Eigen::MatrixXd proj = matrix * point;
//             proj = proj / proj(2, 0);

//             double py = proj(1, 0);
//             double px = proj(0, 0);

//             wx[1] = px - std::floor(px);
//             wx[0] = 1.0 - wx[1];

//             wy[1] = py - std::floor(py);
//             wy[0] = 1.0 - wy[1];

//             int x = static_cast<int>(std::floor(px));
//             int y = static_cast<int>(std::floor(py));

//             if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
//             {
//                 pixel += wx[0] * wy[0] * image.ptr<cv::Vec3b>(y)[x];
//                 ratio += wx[0] * wy[0];
//             }
//             if (x + 1 >= 0 && x + 1 < image.cols && y >= 0 && y < image.rows)
//             {
//                 pixel += wx[1] * wy[0] * image.ptr<cv::Vec3b>(y)[x + 1];
//                 ratio += wx[1] * wy[0];
//             }
//             if (x >= 0 && x < image.cols && y + 1 >= 0 && y + 1 < image.rows)
//             {
//                 pixel += wx[0] * wy[1] * image.ptr<cv::Vec3b>(y + 1)[x];
//                 ratio += wx[0] * wy[1];
//             }
//             if (x + 1 >= 0 && x + 1 < image.cols && y + 1 >= 0 && y + 1 < image.rows)
//             {
//                 pixel += wx[1] * wy[1] * image.ptr<cv::Vec3b>(y + 1)[x + 1];
//                 ratio += wx[1] * wy[1];
//             }
//             // 여기를 더 옮겨줘야 하나??
//             cv::Vec3b val = pixel / ratio + cv::Vec3b(0.5, 0.5, 0.5);
//             if (canvas.ptr<cv::Vec3b>(r)[c + overlap] == cv::Vec3b(0, 0, 0))
//             {
//                 canvas.ptr<cv::Vec3b>(r)[c + overlap] = val;
//             }
//         }
//     }
//     cv::imshow("canvas", canvas);
//     cv::waitKey(0);
// }

// std::tuple<cv::Mat, int, int> crop(cv::Mat &cyl_pic)
// {
//     cv::Mat gray_r;
//     cv::cvtColor(cyl_pic, gray_r, cv::COLOR_BGR2GRAY);

//     int start_row = gray_r.rows, end_row = -1;
//     for (int c = 0; c < gray_r.cols; ++c)
//     {
//         int min_pos = gray_r.rows, max_pos = -1;
//         for (int r = 0; r < gray_r.rows; ++r)
//         {
//             if (static_cast<int>(gray_r.ptr<uchar>(r)[c]) != 0)
//             {
//                 min_pos = std::min(min_pos, r);
//                 max_pos = std::max(max_pos, r);
//             }
//         }

//         start_row = std::min(start_row, min_pos);
//         end_row = std::max(end_row, max_pos);
//     }

//     cv::Mat crp_pic = cv::Mat(end_row - start_row + 1, cyl_pic.cols, CV_8UC3, cv::Scalar::all(255));
//     for (int c = 0; c < cyl_pic.cols; ++c)
//     {
//         for (int r = start_row; r <= end_row; ++r)
//             crp_pic.ptr<cv::Vec3b>(r - start_row)[c] = cyl_pic.ptr<cv::Vec3b>(r)[c];
//     }

//     cv::Mat gray_c;
//     cv::cvtColor(crp_pic, gray_c, cv::COLOR_BGR2GRAY);

//     int start_col = gray_r.cols, end_col = -1;

//     for (int r = 0; r < gray_c.rows; ++r)
//     {
//         int min_pos = gray_r.cols, max_pos = -1;
//         for (int c = 0; c < gray_c.cols; ++c)
//         {
//             if (static_cast<int>(gray_c.ptr<uchar>(r)[c]) != 0)
//             {
//                 min_pos = std::min(min_pos, c);
//                 max_pos = std::max(max_pos, c);
//             }
//         }

//         start_col = std::min(start_col, min_pos);
//         end_col = std::max(end_col, max_pos);
//     }

//     cv::Mat res = cv::Mat(crp_pic.rows, end_col - start_col + 1, CV_8UC3, cv::Scalar::all(255));
//     for (int r = 0; r < crp_pic.rows; ++r)
//     {
//         for (int c = start_col; c <= end_col; ++c)
//             res.ptr<cv::Vec3b>(r)[c - start_col] = crp_pic.ptr<cv::Vec3b>(r)[c];
//     }
//     return std::tuple{res, start_row, start_col};
// }
