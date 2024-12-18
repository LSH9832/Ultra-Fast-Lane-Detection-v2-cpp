#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


class LaneDetector
{
public:
    LaneDetector() {}
    LaneDetector(std::string config_file);

    void setCutHeightRatio(float value);

    void preprocess(cv::Mat& oriImage, float& ratio);

    void detect(cv::Mat& image, std::vector<std::vector<std::vector<float>>>& lines);

    bool isInit();

    std::vector<void*> infer();

    void postprocess(const std::vector<void*> preds, std::vector<std::vector<std::vector<float>>>& lines, float ratio);

    struct Impl;

private:
    Impl* impl_=nullptr;
};






#endif