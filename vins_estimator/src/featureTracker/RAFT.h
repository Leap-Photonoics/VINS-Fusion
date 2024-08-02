#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "../utility/tic_toc.h"

using namespace std;

class RAFT
{
    public:
        RAFT();
        void track(const cv::Mat &img0, const cv::Mat &img1, const vector<cv::Point2f> &prev_pts, vector<cv::Point2f> &curr_pts);
    private:
        torch::jit::script::Module module;
};