#include <iostream>
#include <cmath>

#include "detect.h"
#include "process.h"
// #include "opencv4/opencv2/opencv.hpp"
// #include "opencv4/opencv2/stitching.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"


#if 0 // 양방0
int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout << "put a path of images" << std::endl;
        std::cout << "default directory is \"build\"" << std::endl;
        return 1;
    }
    else if (argc > 2)
    {
        std::cout << "Too many arguments. Only one available" << std::endl;
        return 1;
    }

    std::vector<std::string> fileNames = {};
    cv::glob(argv[1], fileNames);

    std::shared_ptr<Detect> detect;

    auto mid = static_cast<int>((fileNames.size() - 1) / 2);

    std::cout << mid << std::endl;


    cv::Mat src = cv::imread(fileNames[mid], cv::IMREAD_ANYCOLOR);

    cv::Mat resizedSrc, resizedTarget; // 원본 사이즈 : [4032 x 3024]
    cv::Size size(800, 600);
    cv::resize(src, resizedSrc, size);

    warpCylinder(resizedSrc);
    cropBlackArea(resizedSrc, "cylinder");

    // 오른쪽부터
    for(int fileNum = mid + 1; fileNum < fileNames.size(); ++fileNum)
    {
        cv::Mat target = cv::imread(fileNames[fileNum], cv::IMREAD_ANYCOLOR);

        cv::resize(target, resizedTarget, size);
        processImages(resizedSrc, resizedTarget);
        warpCylinder(resizedTarget);
        cropBlackArea(resizedTarget, "cylinder");

        detect = std::make_shared<Detect>(resizedSrc, resizedTarget);
        detect->detectFeatures();
        detect->matchFeatures();
        detect->findMatchingMatrix();
        detect->transformImage("right");

        resizedSrc = detect->getSrcImage();

    }

    for(int fileNum = mid - 1; 0 <= fileNum ; --fileNum)
    {
        cv::Mat target = cv::imread(fileNames[fileNum], cv::IMREAD_ANYCOLOR);

        cv::resize(target, resizedTarget, size);
        processImages(resizedSrc, resizedTarget);
        warpCylinder(resizedTarget);
        cropBlackArea(resizedTarget, "cylinder");

        detect = std::make_shared<Detect>(resizedSrc, resizedTarget);
        detect->detectFeatures();
        detect->matchFeatures();
        detect->findMatchingMatrix();
        detect->transformImage("left");

        resizedSrc = detect->getSrcImage();
    }

    cv::Mat result = detect->getSrcImage();
    cv::imshow("Result", result);
    cv::waitKey(0);
}


#elif 1 // 한방향

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout << "put a path of images" << std::endl;
        std::cout << "default directory is \"build\"" << std::endl;
        return 1;
    }
    else if (argc > 2)
    {
        std::cout << "Too many arguments. Only one available" << std::endl;
        return 1;
    }

    std::vector<std::string> fileNames = {};
    cv::glob(argv[1], fileNames);


    std::shared_ptr<Detect> detect;
    cv::Mat src = cv::imread(fileNames[0], cv::IMREAD_ANYCOLOR);

    cv::Mat resizedSrc, resizedTarget; // 원본 사이즈 : [4032 x 3024]
    cv::Size size(800, 600);

    // cv::imshow("tmp",resizedSrc);
    // cv::waitKey(0);
    cv::resize(src, resizedSrc, size);

    warpCylinder(resizedSrc);
    cv::imshow("g",resizedSrc);
    cv::waitKey(0);
    cropBlackArea11(resizedSrc, "cylinder");

    // 오른쪽부터
    for(int fileNum = 1; fileNum < fileNames.size(); ++fileNum)
    {
        cv::Mat target = cv::imread(fileNames[fileNum], cv::IMREAD_ANYCOLOR);

        cv::resize(target, resizedTarget, size);
        processImages(resizedSrc, resizedTarget);
        warpCylinder(resizedTarget);
        cropBlackArea11(resizedTarget, "cylinder");

        detect = std::make_shared<Detect>(resizedSrc, resizedTarget);
        detect->detectFeatures();
        detect->matchFeatures();
        detect->findMatchingMatrix();
        detect->transformImage("right");

        resizedSrc = detect->getSrcImage();

    }

    cv::Mat result = detect->getSrcImage();
    // cv::imshow("Result", result);
    // cv::waitKey(0);
}


#elif 0
int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cout << "put a path of images" << std::endl;
        std::cout << "default directory is \"build\"" << std::endl;
        return 1;
    }
    else if (argc > 2)
    {
        std::cout << "Too many arguments. Only one available" << std::endl;
        return 1;
    }

    std::vector<std::string> fileNames = {};
    cv::glob(argv[1], fileNames);

    std::vector<cv::Mat> images = {};

    cv::Mat image;
    cv::Size size(1000, 750);
    for(int fileNum = 0; fileNum < fileNames.size(); ++fileNum)
    {
        cv::Mat image = cv::imread(fileNames[fileNum], cv::IMREAD_COLOR);
        cv::resize(image, image, size);

        if(image.empty())
        {
            std::cout << "Image is empty!\n";
            return 1;
        }

        images.emplace_back(image);
    }
    cv::Mat panoramaImage;

    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    cv::Stitcher::Status status = stitcher->stitch(images, panoramaImage);

    if (status != cv::Stitcher::OK)
    {
        std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
        return EXIT_FAILURE;
    }

    cv::imshow("panorama", panoramaImage);
    cv::waitKey(0);
}
#endif
