#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "process.h"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/warpers.hpp"

# define M_PIl 3.141592653589793238462643383279502884L


void processImages(const cv::Mat& srcImage, cv::Mat& targetImage)
{

    auto srcMean = cv::mean(srcImage)*1.05;
    auto targetMean = cv::mean(targetImage)*0.95;

    auto diff = srcMean - targetMean;

    targetImage += diff;
}


void cropBlackArea(cv::Mat& image, const std::string& direction)
{
    // 이미지의 width는 800이니, 0부터 탐색하기 보다는 image.cols - 400부터 하면 될듯.

    if(direction == "right")
    {
        int maxX{-90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; x < image.cols; ++x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                maxX = std::max(maxX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(0,maxX));
    }

    else if(direction == "left")
    {
        int minX{90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; 0 <= x; --x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                minX = std::min(minX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(minX,image.cols));
        // cv::imshow("image",image);
        // cv::waitKey(0);
        // cv::destroyWindow("image");
    }
    else
    {
        int maxX{-90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; x < image.cols; ++x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                maxX = std::max(maxX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(0,maxX));


        int minX{90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; 0 <= x; --x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                minX = std::min(minX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(minX,image.cols));
    }

    cv::imshow("cut", image);
    cv::waitKey(0);
    cv::destroyWindow("cut");

}

void warpCylinder(cv::Mat& image)
{
    // cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    // double k[] = {718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1};
    // cv::Mat K(3,3, CV_32F, k);

    // cv::detail::CylindricalWarper warper(2.0);

    cv::Mat dst(image.rows, image.cols, CV_8UC3);

    // cv::Point prt = warper.warp(image, K, R, 1, 1, dst);
    // std::cout << "here" << std::endl;

    // cv::imshow("dst", dst);
    // cv::waitKey(0);

	float w = image.cols;
	float h = image.rows;
	float f = (w/2)/atan(M_PIl/8);

	for (int i = 0; i <image.rows; i++)
	{
		for (int j = 0; j <image.cols; j++)
		{
			float x = j;
			float y = i;
			float x1 = f * atan((x-w/2)/f) + f * atan(w/(2.0f * f));
			float y1 = f * (y-h/2.0f)/sqrt((x-w/2.0f) * (x-w/2.0f) + f * f) + h/2.0f;

			int col = (int)(x1 + 0.5f);//Add 0.5 for rounding
			int row = (int)(y1 + 0.5f);//Add 0.5 for rounding

			if (col <image.cols && row <image.rows)
			{
				dst.at<cv::Vec3b>(row, col)[0] = image.at<cv::Vec3b>(i, j)[0];
				dst.at<cv::Vec3b>(row, col)[1] = image.at<cv::Vec3b>(i, j)[1];
				dst.at<cv::Vec3b>(row, col)[2] = image.at<cv::Vec3b>(i, j)[2];
			}
		}
    }

    image = dst;
}

void cropBlackArea11(cv::Mat& image, const std::string& direction)
{
    // 이미지의 width는 800이니, 0부터 탐색하기 보다는 image.cols - 400부터 하면 될듯.

    if(direction == "cylinder")
    {
        int maxX{-90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; x < image.cols; ++x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                maxX = std::max(maxX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(0,maxX));


        int minX{90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; 0 <= x; --x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                minX = std::min(minX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(minX,image.cols));
    }
    else
    {
#if 0 //min
        int minX{90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols; (image.cols - image.cols/2) + 50 <= x; --x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                minX = std::min(minX, x);
                break;
            }
        }

        std::cout << "minX : " << minX << std::endl;
        image = image(cv::Range(0,image.rows), cv::Range(0, minX));

#elif 1 // max
        int maxX{-90000000};

        for(int y = 0; y < image.rows; ++y)
        {

            for(int x = image.cols - image.cols/2; x < image.cols; ++x)
            {
                if(image.ptr<cv::Vec3b>(y)[x][0] == 0 && image.ptr<cv::Vec3b>(y)[x][1] == 0 && image.ptr<cv::Vec3b>(y)[x][2] == 0)
                {
                    continue;
                }
                maxX = std::max(maxX, x);
            }
        }

        image = image(cv::Range(0,image.rows), cv::Range(0,maxX));
#endif
    }

    // cv::imshow("cut", image);
    // cv::waitKey(0);
    // cv::destroyWindow("cut");

}
