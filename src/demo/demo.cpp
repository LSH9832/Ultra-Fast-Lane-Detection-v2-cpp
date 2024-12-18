#ifndef LANE_DETECT_DEMO_CPP
#define LANE_DETECT_DEMO_CPP

#include "../../include/laneDetector.h"
#include "../../include/pylike/logger.h"
#include "./color.h"


argparser::ArgumentParser getArgs(int argc, char** argv)
{
    argparser::ArgumentParser parser("lane detector demo argument parser", argc, argv);
    parser.add_option<std::string>("-c", "--config", "config path", "");
    parser.add_option<std::string>("-s", "--source", "video source", "");
    parser.add_option<double>("-r", "--ratio", "cut ratio", 0.5);
    parser.add_help_option();
    return parser.parse();
}


int main(int argc, char** argv)
{
    auto args = getArgs(argc, argv);

    LaneDetector detector(args.get_option_string("--config"));
    if (!detector.isInit())
    {
        ERROR << "lane detector not init!" << ENDL;
        return -1;
    }
    detector.setCutHeightRatio(args.get_option_double("--ratio"));

    cv::VideoCapture cap(args.get_option_string("--source"));
    if (!cap.isOpened())
    {
        ERROR << "can not open '" << args.get_option_string("--source") << "'" << ENDL;
        return -1;
    }

    cv::Mat frame;
    std::vector<std::vector<std::vector<float>>> lines;
    int delay = 1;

    cv::namedWindow("line detect demo", 0);
    cv::resizeWindow("line detect demo", 432, 242);
    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            WARN << "frame is empty, exit." << ENDL;
            break;
        }
        // cv::resize(frame, frame, cv::Size(1600, 900));
        lines.clear();
        detector.detect(frame, lines);

        int colori = 0;
        for (auto line: lines)
        {
            for (auto point: line)
            {
                cv::circle(frame, cv::Point((int)point[0], (int)point[1]), 5, cv::color.as<cv::Scalar>(colori), -1);
            }
            colori+=2;
        }

        cv::imshow("line detect demo", frame);
        int key = cv::waitKey(delay);
        if (key == 27)
        {
            break;
        }
        else if (key == ' ')
        {
            delay = 1 - delay;
        }
    }

    cv::destroyAllWindows();
    cap.release();
    return 0;
}




#endif