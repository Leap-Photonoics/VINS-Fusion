/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern double ENC_N; // 轮速计噪声方差

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Matrix3d RIO; // 轮速计到IMU外参R
extern Eigen::Vector3d TIO_L, TIO_R; // 轮速计到IMU外参T
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern std::string ENCODER_TOPIC; // 轮速计topic
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int NUM_OF_CAM_UNIT;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern vector<std::string> IMAGE_TOPICS;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;

extern bool GNSS_ENABLE;
extern std::string GNSS_EPHEM_TOPIC;
extern std::string GNSS_GLO_EPHEM_TOPIC;
extern std::string GNSS_MEAS_TOPIC;
extern std::string GNSS_IONO_PARAMS_TOPIC;
extern std::string GNSS_TP_INFO_TOPIC;
extern std::vector<double> GNSS_IONO_DEFAULT_PARAMS;
extern bool GNSS_LOCAL_ONLINE_SYNC;
extern std::string LOCAL_TRIGGER_INFO_TOPIC;
extern double GNSS_LOCAL_TIME_DIFF;
extern double GNSS_ELEVATION_THRES;
extern double GNSS_PSR_STD_THRES;
extern double GNSS_DOPP_STD_THRES;
extern uint32_t GNSS_TRACK_NUM_THRES;
extern double GNSS_DDT_WEIGHT;
extern std::string GNSS_RESULT_PATH;

extern bool ENCODER_ENABLE; // 是否融合轮速计

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
