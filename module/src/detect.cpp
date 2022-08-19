#include <iostream>

#include "opencv2/opencv.hpp"
#include "detect.h"


Detect::Detect(cv::Mat& src, cv::Mat target)
: mSrcImage(src), mTargetImage(target)
{};

cv::Mat Detect::getSrcImage()
{
    return mSrcImage;
}


void Detect::detectFeatures()
{

#if 1
    cv::Ptr<cv::Feature2D> extractor = cv::AKAZE::create();
#elif 0
    cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create(1900);
#endif
    extractor->detectAndCompute(mSrcImage, cv::Mat(), mvSrcKeypoint, mSrcDesc);
    extractor->detectAndCompute(mTargetImage, cv::Mat(), mvTargetKeypoint, mTargetDesc);
};


void Detect::matchFeatures()
{
#if 1
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING); // 바이너리 디스크립터용
#elif 0
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2); // SIFT 계열용
#endif

#if 1
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(mSrcDesc, mTargetDesc, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (auto& m : matches)
    {
        if (m[0].distance / m[1].distance < 0.8)
        {
            goodMatches.push_back(m[0]);
        }
    }
#elif 0
    std::vector<cv::DMatch> matches;
    matcher->match(mSrcDesc, mTargetDesc, matches);

    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> goodMatches(matches.begin(), matches.begin() + 300);

#elif 0
	cv::Ptr<cv::FlannBasedMatcher> Matcher = cv::FlannBasedMatcher::create();

    std::vector<std::vector<cv::DMatch>> matches;
    Matcher->knnMatch(mSrcDesc, mTargetDesc, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (auto m : matches)
    {
        if (m[0].distance / m[1].distance < 0.7)
        {
            goodMatches.push_back(m[0]);
        }
    }


	// std::vector<cv::DMatch> matches;
	// Matcher->match(mSrcDesc, mTargetDesc, matches);

    // std::sort(matches.begin(), matches.end());
    // std::vector<cv::DMatch> goodMatches(matches.begin(), matches.begin()+300);
#endif

    std::vector<cv::Point2d> SrcGoodVec = {};
    std::vector<cv::Point2d> TargetGoodVec = {};


    for (size_t i = 0; i < goodMatches.size(); i++)
    {
        SrcGoodVec.push_back(mvSrcKeypoint[goodMatches[i].queryIdx].pt);
        TargetGoodVec.push_back(mvTargetKeypoint[goodMatches[i].trainIdx].pt);
    }

    mvSrcGood = SrcGoodVec;
    mvTargetGood = TargetGoodVec;

    cv::Mat dst;
    cv::drawMatches(mSrcImage, mvSrcKeypoint, mTargetImage, mvTargetKeypoint, goodMatches, dst);
    // cv::imshow("dst",dst);
    // cv::waitKey(0);
    // cv::destroyWindow("dst");
}

void Detect::findMatchingMatrix()
{
    cv::Mat homography = cv::findHomography(mvTargetGood, mvSrcGood, cv::RANSAC);
    mHomoGraphy = homography;
}

void Detect::transformImage(const std::string& direction)
{
#if 0
    for(int y = 0; y < mSrcImage.rows; ++y)
    {
        for(int x = 0; x < mSrcImage.cols; ++x)
        {
            panoramaImage.ptr<int>(y)[x] = mSrcImage.ptr<int>(y)[x];
        }
    }
#elif 1

    cv::Mat panoramaImage;
    auto row = mSrcImage.rows;
    auto col = mTargetImage.cols + mSrcImage.cols;
    cv::Size size(col, row);

    cv::warpPerspective(mTargetImage, panoramaImage, mHomoGraphy, size, cv::INTER_LANCZOS4); // 오른쪽으로만 크기가 커지는 문제
    cv::imshow("warp", mTargetImage);
    cv::waitKey(0);
    cv::destroyWindow("warp");

    cropBlackArea11(panoramaImage, direction);
    // cropBlackArea(panoramaImage, direction);

    cv::imshow("warp", mTargetImage);
    cv::waitKey(0);
    cv::destroyWindow("warp");

    if(direction == "right")
    {
        cv::Mat crop = mSrcImage(cv::Range(0,mSrcImage.rows), cv::Range(0,mSrcImage.cols));
        cv::Mat roi(panoramaImage, cv::Rect(0, 0, crop.cols, crop.rows));

        crop.copyTo(roi); // roi가 panoramaImage의 레퍼런스 인 것 같다. roi에 붙여넣기 하면 panoramaImage에 함께 붙여넣기 됨
        cv::imshow("panorama", panoramaImage);
        cv::waitKey(0);
        cv::destroyWindow("panorama");
        // cropBlackArea11(panoramaImage, direction);
        // cv::imshow("crop-panorama", panoramaImage);
        // cv::waitKey(0);
        // cv::destroyWindow("crop-panorama");

    }
    else
    {
        cv::Mat crop = mSrcImage(cv::Range(0,mSrcImage.rows), cv::Range(0,mSrcImage.cols));
        // cv::imshow("crop",crop);
        // cv::waitKey(0);
        cv::Mat roi(panoramaImage, cv::Rect(100, 0, crop.cols, crop.rows));
        // cv::imshow("roi",roi);
        // cv::waitKey(0);

        crop.copyTo(roi); // roi가 panoramaImage의 레퍼런스 인 것 같다. roi에 붙여넣기 하면 panoramaImage에 함께 붙여넣기 됨
    }


#endif

    mSrcImage = panoramaImage;
}


