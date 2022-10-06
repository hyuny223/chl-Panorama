#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>

#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "compute.hpp"
#include "process.hpp"
#include "lm.hpp"
#include "RANSAC.hpp"
#include "homography.hpp"
#include "image_mosaic.hpp"

#define MP std::make_pair
#define F first
#define S second

int imgcnt = 0;

double f = 718.856;
// double f = 710.0; // default. 사진 마다 다르다.
// double f = 760.0;

int dr[4] = {1, -1, 0, 0};
int dc[4] = {0, 0, 1, -1};

ImageMosaic::ImageMosaic(const std::vector<std::string> &names)
{
    images_num_ = names.size();
    mid_ = static_cast<int>(images_num_ / 2);
    names_ = names;

    images_.reserve(images_num_);
    images_cols_.resize(images_num_);
    images_rows_.resize(images_num_);
    overlap_.resize(images_num_);
}

void ImageMosaic::mosaic()
{
    for (int i = 0; i < names_.size() - 1; ++i)
    {
        std::string key = "H" + std::to_string(i) + std::to_string(i + 1);

        cv::Mat img1 = cv::imread(names_[i], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(names_[i + 1], cv::IMREAD_COLOR);
        cv::resize(img1, img1, img1.size() / 6);
        cv::resize(img2, img2, img2.size() / 6);

        // 0. Cylindrical Warping
        warpCrop(img1, f);
        warpCrop(img2, f);

        if (i == 0)
        {
            images_rows_.at(i) = img1.rows;
            images_rows_.at(i + 1) = img2.rows;

            images_cols_.at(i) = img1.cols;
            images_cols_.at(i + 1) = img2.cols;

            images_.emplace_back(img1);
            images_.emplace_back(img2);
        }
        else
        {
            images_rows_.at(i + 1) = img2.rows;
            images_cols_.at(i + 1) = img2.cols;
            images_.emplace_back(img2);
        }

        // 1. DETECTION
        auto [keys1, desc1, keys2, desc2] = detect(img1, img2);
        auto [good1, good2] = matching(img1, keys1, desc1, img2, keys2, desc2);

        // 2. RANSAC
        Homography h(good1, good2); // good2 -> good1
        RANSAC<Homography> ransac(h, good1.size());
        auto m = ransac.run();
        auto ransac_matrix = m.model();
        double ransac_error = m.error_;
        ransac_matrix /= ransac_matrix(2, 2); // 정규화가 되지 않아서 해준다
        std::vector<int> m_indices = m.getIndices();

        // 3. OPTIMIZATION
        LevenbergMarquardt lm(ransac_matrix);
        lm.setParameters(1e-10, 1e-10, 100, true, ransac_error);

        for (const auto idx : m_indices)
        {
            lm.addObservation(good1[idx].x, good1[idx].y);
            lm.addPoints(good2[idx].x, good2[idx].y);
        }
        lm.run();
        matrices_[key] = lm.model();
    }

    // 4. 중간을 기준으로 이미지 만들기
    sortH(matrices_);

    // 5. stitcing
    stitch(matrices_);
}

void ImageMosaic::sortH(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices)
{
    std::size_t img_num = matrices.size() + 1;

    std::string key = "H" + std::to_string(mid_) + std::to_string(mid_);

    // 0, 1, 2, 3, 4가 있다면  22를 만드는 과정
    matrices[key] = Eigen::Matrix<double, 3, 3>::Identity();

    // 0, 1, 2, 3, 4가 있다면  12, 02를 만드는 과정
    for (int i = mid_ - 1; i >= 0; --i)
    {
        key.clear();
        key = "H" + std::to_string(i) + std::to_string(mid_);
        std::string curr_key = "H" + std::to_string(i + 1) + std::to_string(mid_);
        std::string next_key = "H" + std::to_string(i) + std::to_string(i + 1);

        Eigen::Matrix<double, 3, 3> tmp = matrices[next_key] * matrices[curr_key];
        tmp /= tmp(2, 2); // 정규화를 해야하나? --> 할 필요는 없다. 그러나 뒤에 다시 최적화 해야하면 필요하다.

        matrices[key] = tmp;
    }
    // 0, 1, 2, 3, 4가 있다면  32, 42를 만드는 과정
    for (int i = mid_ + 1; i < img_num; ++i)
    {
        key.clear();
        key = "H" + std::to_string(i) + std::to_string(mid_);
        std::string curr_key = "H" + std::to_string(i - 1) + std::to_string(mid_);
        std::string next_key = "H" + std::to_string(i - 1) + std::to_string(i);

        Eigen::Matrix<double, 3, 3> tmp = matrices[next_key].inverse() * matrices[curr_key];
        tmp /= tmp(2, 2);

        matrices[key] = tmp;
    }
}

void ImageMosaic::stitch(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices)
{
    /*
    1. 캔버스 생성. 이미지 크기에 맞게 생성.
    중간 이미지를 기준으로 와프를 시킨다. 그러면 원래 이미지가 [세로 x 가로] == [50, 50]이었다가 H*images 했더니 [50, 40]이 된다.
    그러면 이동을 왼쪽으로 10만큼 했다는 의미가 된다. 이러한 방식으로 overlap 되는 부분을 구할 수 있다.

    mask : 겹치는 부분 찾기
    offset : 이미지 위치 보정
    */
    cv::Mat canvas, mask;
    Eigen::Matrix<int, 1, 3> offset;
    generateCanvas(matrices, canvas, mask, offset);

    // 가운데 이미지
    std::string key = "H" + std::to_string(mid_) + std::to_string(mid_);
    warpImage(images_.at(mid_), canvas, mask, matrices[key], offset);

    // 왼쪽 스티칭
    for (int i = mid_ - 1; i >= 0; --i)
    {
        std::string key = "H" + std::to_string(i) + std::to_string(mid_);
        warpImage(images_.at(i), canvas, mask, matrices[key], offset);
    }

    // 오른쪽 스티칭
    for (int i = mid_ + 1; i < images_num_; ++i)
    {
        std::string key = "H" + std::to_string(i) + std::to_string(mid_);
        warpImage(images_.at(i), canvas, mask, matrices[key], offset);
    }
    cv::imshow("canvas", canvas);
    while(1)
    {
        if(cv::waitKey() == 27)
        {
            break;
        }
    }

    // 2. 블렌딩 및 그래프 컷 --> 자연스러운 스티칭
}

void ImageMosaic::generateCanvas(std::unordered_map<std::string, Eigen::Matrix<double, 3, 3>> &matrices,
                                 cv::Mat &canvas,
                                 cv::Mat &mask,
                                 Eigen::Matrix<int, 1, 3> &offset)
{
    int inf = std::numeric_limits<int>::max();
    Eigen::Matrix<int, 1, 3> min_crd_canvas = {inf, inf, inf};
    Eigen::Matrix<int, 1, 3> max_crd_canvas = {0, 0, 0};

    for (int i = 0; i < images_num_; ++i)
    {
        std::string key = "H" + std::to_string(i) + std::to_string(mid_);
        Eigen::Matrix<double, 3, 3> matrix = matrices[key];

        Eigen::Matrix<int, 1, 3> min_crd, max_crd;
        computeMove(matrix,
                    images_rows_.at(i),
                    images_cols_.at(i),
                    min_crd,
                    max_crd);

        for (int i = 0; i < 3; ++i)
        {
            min_crd_canvas(0, i) = std::min(min_crd_canvas(0, i), min_crd(0, i));
            max_crd_canvas(0, i) = std::max(max_crd_canvas(0, i), max_crd(0, i));
        }
    }
    int canvas_height = (max_crd_canvas - min_crd_canvas)(0, 1) + 1;
    int canvas_width = (max_crd_canvas - min_crd_canvas)(0, 0) + 1;

    canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    offset = min_crd_canvas;
    offset(0,2) = 0;

    mask = cv::Mat::ones(canvas_height, canvas_width, CV_8UC1) * 255;
}

void ImageMosaic::computeMove(Eigen::Matrix<double, 3, 3> &matrix,
                              const int row,
                              const int col,
                              Eigen::Matrix<int, 1, 3> &min_crd,
                              Eigen::Matrix<int, 1, 3> &max_crd)
{
    Eigen::Matrix<double, 3, 4> pts;
    pts << 0.0, col, col, 0.0,
        0.0, 0.0, row, row,
        1.0, 1.0, 1.0, 1.0;

    Eigen::Matrix<double, 3, 4> cam_crd = matrix * pts;
    cam_crd = cam_crd.array().rowwise() / cam_crd.row(2).array();

    double x_min = std::numeric_limits<double>::infinity(), y_min = std::numeric_limits<double>::infinity();
    double x_max = std::numeric_limits<double>::lowest(), y_max = std::numeric_limits<double>::lowest();

    for (int i = 0; i < 4; ++i)
    {
        x_min = std::min(x_min, cam_crd(0, i));
        y_min = std::min(y_min, cam_crd(1, i));

        x_max = std::max(x_max, cam_crd(0, i));
        y_max = std::max(y_max, cam_crd(1, i));
    }

    min_crd << static_cast<int>(std::floor(x_min)), static_cast<int>(std::floor(y_min)), 1;
    max_crd << static_cast<int>(std::ceil(x_max)), static_cast<int>(std::ceil(y_max)), 1;
}


// cv::Mat fin;
// pic[0].copyTo(fin);

// for (int i = 1; i < imgcnt; i++)
// {
// 	cv::Mat diff(pic[i].rows, cum_xshift[i - 1] + pic[i - 1].cols - cum_xshift[i], CV_32SC1);
// 	for (int r = 0; r < pic[i].rows; ++r)
// 	{
// 		for (int c = 0; c < diff.cols; ++c)
// 		{
// 			cv::Vec3b past = pic[i - 1].ptr<cv::Vec3b>(r)[c + cum_xshift[i] - cum_xshift[i - 1]];
// 			cv::Vec3b now = pic[i].ptr<cv::Vec3b>(r)[c];
// 			int val = std::max(std::max(std::abs(static_cast<int>(past[0]) - static_cast<int>(now[0])),
// 										std::abs(static_cast<int>(past[1]) - static_cast<int>(now[1]))),
// 							   			std::abs(static_cast<int>(past[2]) - static_cast<int>(now[2])));
// 			diff.ptr<int>(r)[c] = val; // 세 채널 중 가장 큰 값
// 		}
// 	}

// 	std::priority_queue<std::pair<int, std::pair<int, int>>> Q;

// 	// "choice" matrix is the decision of which picture to use:
// 	// -1 to use the (i-1)th picture,
// 	//  1 to use the ith picture.

// 	// This matrix only contains the precise intersection of the two image,
// 	// as you can see from the size of this mat.

// 	cv::Mat choice = cv::Mat::zeros(pic[i].rows, cum_xshift[i - 1] + pic[i - 1].cols - cum_xshift[i], CV_32SC1);
// 	for (int r = 0; r < pic[i].rows; ++r) // 이 부분은 무엇??
// 	{
// 		Q.push(MP(1000, MP(-1, r * choice.cols + 0)));
// 		Q.push(MP(1000, MP(-1, r * choice.cols + 1)));
// 		Q.push(MP(1000, MP(-1, r * choice.cols + 2)));
// 		Q.push(MP(1000, MP(-1, r * choice.cols + 3)));
// 		Q.push(MP(1000, MP(-1, r * choice.cols + 4)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 1)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 2)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 3)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 4)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 5)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 6)));
// 		Q.push(MP(1000, MP(1, r * choice.cols + choice.cols - 7)));
// 	}

// 	while (!Q.empty()) // choice 즉, 이어붙일 Mat의 cols기반으로
// 	{
// 		std::pair<int, std::pair<int, int>> one_pix = Q.top(); Q.pop();
// 		int clr = one_pix.S.F;
// 		int r = one_pix.S.S / choice.cols;
// 		int c = one_pix.S.S % choice.cols;

// 		if (choice.ptr<int>(r)[c] != 0)
// 		{
// 			continue;
// 		}
// 		choice.ptr<int>(r)[c] = clr;
// 		// cv::imshow("?", choice);
// 		// cv::waitKey(1);
// 		for (int k = 0; k < 4; k++)
// 		{
// 			int nr = r + dr[k];
// 			int nc = c + dc[k];
// 			if (nr < 0 || nr >= choice.rows || nc < 0 || nc >= choice.cols)
// 			{
// 				continue;
// 			}

// 			if (choice.ptr<int>(nr)[nc] == 0)
// 			{
// 				Q.push(MP(diff.ptr<int>(nr)[nc], MP(clr, nr * choice.cols + nc)));
// 			}
// 		}
// 	}

// 	// Graph cut
// 	for (int r = 0; r < pic[i].rows; ++r)
// 	{
// 		for (int c = 0; c < choice.cols; ++c)
// 		{
// 			if (choice.at<int>(r, c) == 1)
// 			{
// 				fin.at<cv::Vec3b>(r, c + cum_xshift[i]) = pic[i].at<cv::Vec3b>(r, c);

// 				int boundary = 0;
// 				for (int k = 0; k < 4; k++)
// 				{
// 					int nr = r + dr[k];
// 					int nc = c + dc[k];
// 					if (nr < 0 || nr >= choice.rows || nc < 0 || nc >= choice.cols)
// 						continue;

// 					if (choice.at<int>(nr, nc) == -1)
// 						boundary = 1;
// 				}

// 				// For the boundary line
// 				// if(boundary)
// 				//	fin.at<Vec3b>(x, y + cum_xshift[i]) = Vec3b(0, 0, 255);
// 			}
// 		}

// 		for (int c = choice.cols; c < pic[i].cols; c++)
// 			fin.at<cv::Vec3b>(r, c + cum_xshift[i]) = pic[i].at<cv::Vec3b>(r, c);
// 	}
// }
