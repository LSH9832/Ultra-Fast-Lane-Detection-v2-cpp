#ifndef LANE_DETECTOR_COMMON_CPP
#define LANE_DETECTOR_COMMON_CPP

#include "../../include/pylike/np.h"
#include "../../include/pylike/logger.h"
#include <yaml-cpp/yaml.h>
#include <math.h>

#include "../../include/laneDetector.h"

// #define CULANE 1
// // culane
// #if CULANE
// // #define NUM_GRID_ROW 200
// // #define NUM_CLS_ROW 72
// // #define NUM_GRID_COL 100
// // #define NUM_CLS_COL 81
// // #define INPUT_WIDTH 1600
// // #define INPUT_HEIGHT 320
// // #define LOCAL_WIDTH 1
// // #define BATCH 1

// #else
// // tusimple
// #define NUM_GRID_ROW 100
// #define NUM_CLS_ROW 56
// #define NUM_GRID_COL 100
// #define NUM_CLS_COL 41
// #define INPUT_WIDTH 800
// #define INPUT_HEIGHT 320
// #define LOCAL_WIDTH 3
// #define BATCH 1
// #endif



static const std::vector<float> mean_ = {0.485, 0.456, 0.406},
                                std_  = {0.229, 0.224, 0.225};

// static const std::vector<float> mean_ = {0., 0., 0.},
//                                 std_  = {1., 1., 1.};


static void blobFromImagef(cv::Mat& img, float* blob, bool normalize=false)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    float range = normalize?(1./255):1;
    for (size_t c = 0; int(c) < channels; c++)
    {
        int c_ = channels - 1 - c;  // bgr2rgb
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = ((float)(img.at<cv::Vec3b>(h, w)[c_] * range) - mean_[c_]) / std_[c_];
    }
        
}


// ----------------------------- implementation ------------------------------- //

struct LaneDetector::Impl
{
    int NUM_GRID_ROW = 100;
    int NUM_CLS_ROW = 56;
    int NUM_GRID_COL = 100;
    int NUM_CLS_COL = 41;
    int INPUT_WIDTH = 800;
    int INPUT_HEIGHT = 320;
    int LOCAL_WIDTH = 3;
    int NUM_LANES = 4;
    int BATCH = 1;

    std::vector<int> colIdx={}, rowIdx={};
    TimeCount t;

    int cutH=0;
    float cutHRatio=0.0;
    float ratio=1.0, r=0.;
    bool outputFloat=true, isinit_=false, showInfo=false, showInput=false;
    std::vector<float> zps={}, scales={};
    std::vector<float> rowAnchor={}, colAnchor={};

    std::string inputName;
    std::vector<std::string> outputNames;

    struct Custom;

    Custom* cust_=nullptr;

    void* inputs=nullptr;
    std::vector<void*> preds_;

    bool init(std::string config);

    void preds2coords(
        void* locRow_, void* locCol_, 
        void* existRow_, void* existCol_,
        std::vector<std::vector<std::vector<float>>>& lines, 
        int batchID=0
    );
};


void LaneDetector::Impl::preds2coords(
    void* locRow_, void* locCol_, 
    void* existRow_, void* existCol_,
    std::vector<std::vector<std::vector<float>>>& lines, 
    int batchID
)
{
    // batchID++;
    if (outputFloat)
    {
        
        auto locRow = np::Array<float>({BATCH, NUM_GRID_ROW, NUM_CLS_ROW, NUM_LANES}, (float*)locRow_);
        auto locCol = np::Array<float>({BATCH, NUM_GRID_COL, NUM_CLS_COL, NUM_LANES}, (float*)locCol_);
        auto existRow = np::Array<float>({BATCH, 2, NUM_CLS_ROW, NUM_LANES}, (float*)existRow_);
        auto existCol = np::Array<float>({BATCH, 2, NUM_CLS_COL, NUM_LANES}, (float*)existCol_);

        std::vector<std::vector<float>> line;
        std::vector<int> valids(NUM_CLS_ROW);
        int sumValids = 0;
        float tmp=0., softSum = 0., onetmp=0.;
        
        for(int i: rowIdx)
        {
            // valids.clear();
            // valids.resize(NUM_CLS_ROW);
            sumValids = 0;
            for(int k=0;k<NUM_CLS_ROW;k++)
            {
                valids[k] = existRow.argmaxAt(1, {batchID, -1, k, i});
                sumValids += valids[k];
            }

            if (sumValids <= NUM_CLS_ROW / 4) continue;

            line.clear();
            for(int k=0;k<NUM_CLS_ROW;k++)
            {
                if (!valids[k]) continue;

                int idxRow = locRow.argmaxAt(1, {batchID, -1, k, i});

                tmp = 0, softSum = 0.;
                for (int idx=std::max(0, idxRow-LOCAL_WIDTH);
                     idx<std::min(NUM_GRID_ROW-1, idxRow+LOCAL_WIDTH)+1;
                     idx++)
                {
                    onetmp = expf(locRow.at({batchID, idx, k, i}));
                    tmp += idx * onetmp;
                    softSum += onetmp;
                }
                line.push_back({
                    ratio * (((tmp / softSum) + 0.5f) / (NUM_GRID_ROW-1)) * INPUT_WIDTH,
                    ratio * (rowAnchor[k] * INPUT_HEIGHT + cutH)
                });
            }
            lines.push_back(line);
        }
        
        // return;
        valids.clear();
        valids.resize(NUM_CLS_COL);
        for(int i: colIdx)
        {
            
            sumValids = 0;
            for(int k=0;k<NUM_CLS_COL;k++)
            {
                valids[k] = existCol.argmaxAt(1, {batchID, -1, k, i});
                sumValids += valids[k];
            }

            if (sumValids <= NUM_CLS_COL / 4) continue;            

            line.clear();
            for(int k=0;k<NUM_CLS_COL;k++)
            {
                if (!valids[k]) continue;
                
                int idxCol = locCol.argmaxAt(1, {batchID, -1, k, i});

                tmp = 0, softSum = 0.;
                
                for (int idx=std::max(0, idxCol-LOCAL_WIDTH);
                     idx<std::min(NUM_GRID_COL-1, idxCol+LOCAL_WIDTH)+1;
                     idx++)
                {
                    onetmp = expf(locCol.at({batchID, idx, k, i}));
                    tmp += idx * onetmp;
                    softSum += onetmp;
                }
                
                float base = ((tmp / softSum) + 0.5f) / (NUM_GRID_COL-1);
                base = (base - r) / (1.f - r);
                line.push_back({
                    ratio * colAnchor[k] * INPUT_WIDTH,
                    ratio * (base  * INPUT_HEIGHT + cutH)
                });
            }

            
            lines.push_back(line);
        }
        
    }
    else
    {
        ERROR << "do not support not-float data" << ENDL;
    }
    
}


LaneDetector::LaneDetector(std::string config)
{
    logsetStdoutFormat("[$TIME] [$LEVEL] $LOCATION - $MSG");
    if (impl_ == nullptr)
    {
        impl_ = new Impl();
    }
    if(!impl_->init(config))
    {
        ERROR << "failed to init model" << ENDL;
    }
}


void LaneDetector::detect(cv::Mat& image, std::vector<std::vector<std::vector<float>>>& lines)
{
    float ratio=1.0;

    if (impl_->showInfo) impl_->t.tic(0);
    preprocess(image, ratio);
    if (impl_->showInfo) 
    {
        impl_->t.toctic(0, {1});
    }
    auto result = infer();
    if (impl_->showInfo) 
    {
        impl_->t.toctic(1, {2});
    }
    postprocess(result, lines, ratio);
    if (impl_->showInfo) 
    {
        impl_->t.toctic(2, {3});
    }
    INFO << std::fixed << std::setprecision(3) 
         << "pre: " << impl_->t.get_timeval_f(0)
         << "ms, infer: " << impl_->t.get_timeval_f(1)
         << "ms, post: " << impl_->t.get_timeval_f(2) << "ms" << ENDL;
}


bool LaneDetector::isInit()
{
    if (impl_ == nullptr) return false;
    if (!impl_->isinit_) return false;
    if (impl_->inputs == nullptr) return false;
    if (!impl_->preds_.size()) return false;
    return true;
}


void LaneDetector::setCutHeightRatio(float value)
{
    if(impl_ == nullptr)
    {
        ERROR << "line detector not init!" << ENDL;
        return;
    }
    impl_->cutHRatio = value;
}


void LaneDetector::preprocess(cv::Mat& oriImage, float& ratio)
{
    if (!impl_->isinit_)
    {
        ERROR << "lane detector not init!" << ENDL;
        return;
    }
    cv::Mat inputImage;
    // resize, cols -> width, rows->height
    if (oriImage.cols != impl_->INPUT_WIDTH)
    {
        ratio = (float)oriImage.cols / impl_->INPUT_WIDTH;
        cv::resize(oriImage, inputImage, cv::Size(impl_->INPUT_WIDTH, (int)((float)oriImage.rows / ratio)), 0, 0, cv::INTER_NEAREST);
    }
    else
    {
        ratio = 1.;
        inputImage = oriImage;
    }

    cv::Rect roi;
    roi.x = 0, roi.width = impl_->INPUT_WIDTH, roi.height = std::min(impl_->INPUT_HEIGHT, inputImage.rows);
    impl_->cutH = (int)(impl_->cutHRatio * inputImage.rows);
    roi.y = std::max(0, std::min(inputImage.rows - impl_->INPUT_HEIGHT, impl_->cutH));
    impl_->cutH = roi.y;
    // INFO << roi << ENDL;

    inputImage = inputImage(roi);
    if (inputImage.rows < impl_->INPUT_HEIGHT)
    {
        // pad
        cv::copyMakeBorder(inputImage, inputImage, 0, impl_->INPUT_HEIGHT-inputImage.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    }


    if (impl_->showInput) cv::imshow("input", inputImage);

    // nhwc2nchw
    if (impl_->inputs != nullptr)
    {
        blobFromImagef(inputImage, (float*)impl_->inputs, true);
    }
    else
    {
        ERROR << "inputs not init!" << ENDL;
        return;
    }
}


#endif