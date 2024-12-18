#ifndef LANE_DETECTOR_MNN_CPP
#define LANE_DETECTOR_MNN_CPP

#include "./common.cpp"

#include "MNN/MNNDefine.h"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/Interpreter.hpp"


struct LaneDetector::Impl::Custom
{
    std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
    MNN::ScheduleConfig config;            //计划配置
    MNN::BackendConfig* backendConfig = nullptr;
    MNN::Session *session = nullptr;
    MNN::Tensor *inputTensor = nullptr;
    std::vector<MNN::Tensor*> outputTensors;

    MNN::Tensor* inputHostTensor = nullptr;
    std::vector<MNN::Tensor*> outputHostTensors;
};


bool LaneDetector::Impl::init(std::string config)
{
    if (cust_ == nullptr)
    {
        cust_ = new Custom();
    }

    YAML::Node cfg = YAML::LoadFile(config);

    std::string modelPath = cfg["model"].as<std::string>();
    inputName = cfg["input_name"].as<std::string>();
    outputNames = cfg["output_names"].as<std::vector<std::string>>();
    NUM_GRID_ROW = cfg["num_grid_row"].as<int>();
    NUM_GRID_COL = cfg["num_grid_col"].as<int>();
    NUM_CLS_ROW = cfg["num_cls_row"].as<int>();
    NUM_CLS_COL = cfg["num_cls_col"].as<int>();
    INPUT_WIDTH = cfg["input_width"].as<int>();
    INPUT_HEIGHT = cfg["input_height"].as<int>();
    LOCAL_WIDTH = cfg["local_width"].as<int>();
    NUM_LANES = cfg["num_lanes"].as<int>();
    BATCH = cfg["batch"].as<int>();
    r = cfg["ratio"].as<float>();
    showInfo = cfg["show_info"].as<bool>();
    if (cfg["show_input"].IsDefined()) showInput = cfg["show_input"].as<bool>();

    rowAnchor=np::linespace<float>(cfg["h_start"].as<float>(), 1, NUM_CLS_ROW), 
    colAnchor=np::linespace<float>(0, 1, NUM_CLS_COL);

    auto mode = cfg["mode"].as<std::vector<std::vector<int>>>();

    if (!mode[0].size())
    {
        rowIdx = {};
    }
    else if (mode[0][0] == -1)
    {
        rowIdx.clear();
        for(int i=0;i<NUM_LANES;i++)
        {
            rowIdx.push_back(i);
        }
    }
    else
    {
        rowIdx = mode[0];
    }
    
    if (!mode[1].size())
    {
        colIdx = {};
    }
    else if (mode[1][0] == -1)
    {
        colIdx.clear();
        for(int i=0;i<NUM_LANES;i++)
        {
            colIdx.push_back(i);
        }
    }
    else
    {
        colIdx = mode[1];
    }

    

    // mnn do something

    if (cust_->backendConfig == nullptr)
    {
        cust_->backendConfig = new MNN::BackendConfig();
        cust_->config.backendConfig = cust_->backendConfig;
    }
    cust_->net = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(
            osp::join({osp::dirname(config), modelPath}).c_str()
        )
    );
    cust_->config.backendConfig->precision = MNN::BackendConfig::Precision_Normal;
    cust_->config.backendConfig->power = MNN::BackendConfig::Power_Normal;
    cust_->config.backendConfig->memory = MNN::BackendConfig::Memory_Normal;
    cust_->config.type = MNN_FORWARD_CUDA;
    cust_->session = cust_->net->createSession(cust_->config);

    auto outputs = cust_->net->getSessionOutputAll(cust_->session);

    //获取输入输出tensor
    cust_->inputTensor = cust_->net->getSessionInput(cust_->session, NULL);
    cust_->outputTensors.resize(outputNames.size());
    for (int i=0;i<outputNames.size();i++)
    {
        cust_->outputTensors[i] = outputs[outputNames[i]];
    }

    isinit_ = false;
    auto shapes = std::vector<std::vector<int>>{
        {BATCH, NUM_GRID_ROW, NUM_CLS_ROW, NUM_LANES},
        {BATCH, NUM_GRID_COL, NUM_CLS_COL, NUM_LANES},
        {BATCH, 2, NUM_CLS_ROW, NUM_LANES},
        {BATCH, 2, NUM_CLS_COL, NUM_LANES}
    };

    // check
    for (int k=0;k<outputNames.size();k++)
    {
        if(cust_->outputTensors[k]->shape() != shapes[k])
        {
            std::ostringstream oss1, oss2;
            auto shape = cust_->outputTensors[k]->shape();
            for (int i=0;i<shape.size();i++)
            {
                if(i) oss1 << "x";
                oss1 << shape[i];
            }

            for(int i=0;i<shapes[k].size();i++)
            {
                if(i) oss2 << "x";
                oss2 << shapes[k][i];
            }

            ERROR << "output '" << outputNames[k] << "' should be " << oss1.str() << " but got " << oss2.str() << "." << ENDL;
            return isinit_;
        }
        
    }

    cust_->inputHostTensor = new MNN::Tensor(cust_->inputTensor, MNN::Tensor::CAFFE);
    inputs = cust_->inputHostTensor->host<void>();

    cust_->outputHostTensors.resize(outputNames.size());
    preds_.resize(outputNames.size());
    for(int i=0;i<cust_->outputTensors.size();i++)
    {
        cust_->outputHostTensors[i] = new MNN::Tensor(cust_->outputTensors[i], MNN::Tensor::CAFFE);
        preds_[i] = cust_->outputHostTensors[i]->host<void>();
    }

    isinit_ = true;
    return isinit_;
}


std::vector<void*> LaneDetector::infer()
{
    if (!impl_->isinit_)
    {
        ERROR << "lane detector not init!" << ENDL;
        return {};
    }

    if (impl_->outputFloat)
    {
        // auto t0 = pytime::time();
        impl_->cust_->inputTensor->copyFromHostTensor(impl_->cust_->inputHostTensor);
        // dt = pytime::time() - t0;
        // std::cout << " copy2: " << dt * 1000 << "ms";

        // t0 = pytime::time();
        impl_->cust_->net->runSession(impl_->cust_->session);
        // dt = pytime::time() - t0;
        // std::cout << " infer: " << dt * 1000 << "ms";
        // pytime::sleep(1);

        // t0 = pytime::time();
        for (int i=0;i<impl_->outputNames.size();i++)
        {
            impl_->cust_->outputTensors[i]->copyToHostTensor(impl_->cust_->outputHostTensors[i]);
            // impl_->preds_[i] = impl_->cust_->outputHostTensors[i]->host<void>();
        }
        return impl_->preds_;
    }
    else
    {
        ERROR << "only support float data now." << ENDL;
        return {};
    }
}

void LaneDetector::postprocess(const std::vector<void*> preds, std::vector<std::vector<std::vector<float>>>& lines, float ratio)
{
    if (ratio > 0) impl_->ratio = ratio;
    // INFO << impl_->ratio << ENDL;
    impl_->preds2coords(preds[0], preds[1], preds[2], preds[3], lines, 0);
}


#endif