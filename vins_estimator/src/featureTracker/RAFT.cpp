#include "RAFT.h"

RAFT::RAFT()
{
    module = torch::jit::load("/workspaces/vins-fusion/thirdparty/RAFT/model.pt");
    module.to(torch::kCUDA);
}

void RAFT::track(const cv::Mat &img0, const cv::Mat &img1, const vector<cv::Point2f> &prev_pts, vector<cv::Point2f> &curr_pts)
{
    static fstream log("/workspaces/vins-fusion/thirdparty/RAFT/ref_time.txt", ios::out);
    ROS_ASSERT(curr_pts.empty());
    TicToc tic;
    torch::Tensor tensor0 = torch::from_blob(img0.data, {1, 1, img0.rows, img0.cols}, torch::kUInt8).to(torch::kCUDA), 
                  tensor1 = torch::from_blob(img1.data, {1, 1, img1.rows, img1.cols}, torch::kUInt8).to(torch::kCUDA);
    log << tic.toc() << " ";
    auto output = module.forward({tensor0, tensor1}).toTuple();
    auto flow = output->elements()[1].toTensor().to(torch::kCPU);
    log << tic.toc() << " ";
    for (auto pt: prev_pts)
    {
        int ix = (int)pt.x, iy = (int)pt.y;
        double wx = pt.x - ix, wy = pt.y - iy;
        double w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy), w10 = (1 - wx) * wy, w11 = wx * wy;
        float dx = w00 * flow[0][0][iy][ix].item<float>() + w01 * flow[0][0][iy][ix + 1].item<float>() + w10 * flow[0][0][iy + 1][ix].item<float>() + w11 * flow[0][0][iy + 1][ix + 1].item<float>(),
              dy = w00 * flow[0][1][iy][ix].item<float>() + w01 * flow[0][1][iy][ix + 1].item<float>() + w10 * flow[0][1][iy + 1][ix].item<float>() + w11 * flow[0][1][iy + 1][ix + 1].item<float>();
        cv::Point2f new_pt = pt + cv::Point2f(dx, dy);
        curr_pts.push_back(new_pt);
    }
    log << tic.toc() << endl;
}