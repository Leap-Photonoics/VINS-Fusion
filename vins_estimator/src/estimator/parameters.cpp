/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double ENC_N;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int NUM_OF_CAM_UNIT;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
vector<std::string> IMAGE_TOPICS;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

bool GNSS_ENABLE;
std::string GNSS_EPHEM_TOPIC;
std::string GNSS_GLO_EPHEM_TOPIC;
std::string GNSS_MEAS_TOPIC;
std::string GNSS_IONO_PARAMS_TOPIC;
std::string GNSS_TP_INFO_TOPIC;
std::vector<double> GNSS_IONO_DEFAULT_PARAMS;
bool GNSS_LOCAL_ONLINE_SYNC;
std::string LOCAL_TRIGGER_INFO_TOPIC;
double GNSS_LOCAL_TIME_DIFF;
double GNSS_ELEVATION_THRES;
double GNSS_PSR_STD_THRES;
double GNSS_DOPP_STD_THRES;
uint32_t GNSS_TRACK_NUM_THRES;
double GNSS_DDT_WEIGHT;
std::string GNSS_RESULT_PATH;

bool ENCODER_ENABLE; // 是否融合轮速计
std::string ENCODER_TOPIC;
Eigen::Matrix3d RIO; // 轮速计到IMU外参R
Eigen::Vector3d TIO_L, TIO_R; // 轮速计到IMU外参T

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        int infl_w = fsSettings["infl_w"], infl_n = fsSettings["infl_n"];
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        ACC_N *= infl_n;
        GYR_N *= infl_n;
        ACC_W *= infl_w;
        GYR_W *= infl_w;
        ROS_INFO("acc_n %f acc_w %f gyr_n %f gyr_w %f\n", ACC_N, ACC_W, GYR_N, GYR_W);
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);
    assert(NUM_OF_CAM == 1 || NUM_OF_CAM % 2 == 0);
    STEREO = NUM_OF_CAM != 1;

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    vector<string> cams;
    fsSettings["cams"] >> cams;
    assert(cams.size() == NUM_OF_CAM);
    NUM_OF_CAM_UNIT = NUM_OF_CAM == 1 ? 1 : NUM_OF_CAM / 2;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_INFO_STREAM("cam" << i << " " << cams[i]);
        CAM_NAMES.push_back(configPath + "/" + cams[i] + ".yaml");
        cv::FileStorage fs(CAM_NAMES[i], cv::FileStorage::READ);
        string image_topic;
        fs["image_topic"] >> image_topic;
        IMAGE_TOPICS.push_back(image_topic);
        cv::Mat cv_T;
        fs["TIC"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
        fs.release();
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    int gnss_enable_value = fsSettings["gnss_enable"];
    GNSS_ENABLE = (gnss_enable_value == 0 ? false : true);

    if (GNSS_ENABLE)
    {
        fsSettings["gnss_ephem_topic"] >> GNSS_EPHEM_TOPIC;
        fsSettings["gnss_glo_ephem_topic"] >> GNSS_GLO_EPHEM_TOPIC;
        fsSettings["gnss_meas_topic"] >> GNSS_MEAS_TOPIC;
        fsSettings["gnss_iono_params_topic"] >> GNSS_IONO_PARAMS_TOPIC;
        cv::Mat cv_iono;
        fsSettings["gnss_iono_default_parameters"] >> cv_iono;
        Eigen::Matrix<double, 1, 8> eigen_iono;
        cv::cv2eigen(cv_iono, eigen_iono);
        for (uint32_t i = 0; i < 8; ++i)
            GNSS_IONO_DEFAULT_PARAMS.push_back(eigen_iono(0, i));
        
        fsSettings["gnss_tp_info_topic"] >> GNSS_TP_INFO_TOPIC;
        int gnss_local_online_sync_value = fsSettings["gnss_local_online_sync"];
        GNSS_LOCAL_ONLINE_SYNC = (gnss_local_online_sync_value == 0 ? false : true);
        if (GNSS_LOCAL_ONLINE_SYNC)
            fsSettings["local_trigger_info_topic"] >> LOCAL_TRIGGER_INFO_TOPIC;
        else
            GNSS_LOCAL_TIME_DIFF = fsSettings["gnss_local_time_diff"];

        GNSS_ELEVATION_THRES = fsSettings["gnss_elevation_thres"];
        const double gnss_ddt_sigma = fsSettings["gnss_ddt_sigma"];
        GNSS_PSR_STD_THRES = fsSettings["gnss_psr_std_thres"];
        GNSS_DOPP_STD_THRES = fsSettings["gnss_dopp_std_thres"];
        const double track_thres = fsSettings["gnss_track_num_thres"];
        GNSS_TRACK_NUM_THRES = static_cast<uint32_t>(track_thres);
        GNSS_DDT_WEIGHT = 1.0 / gnss_ddt_sigma;
        GNSS_RESULT_PATH = OUTPUT_FOLDER + "/gnss_result.csv";
        // clear output file
        std::ofstream gnss_output(GNSS_RESULT_PATH, std::ios::out);
        gnss_output.close();
        ROS_INFO_STREAM("GNSS enabled");
    }

    ENCODER_ENABLE = (int)fsSettings["encoder_enable"];
    if (ENCODER_ENABLE)
    {
        fsSettings["encoder_topic"] >> ENCODER_TOPIC;
        ENC_N = fsSettings["enc_n"]; // 轮速计噪声方差
        cv::Mat cv_RIO, cv_TIO_L, cv_TIO_R;
        fsSettings["RIO"] >> cv_RIO;
        fsSettings["TIO_L"] >> cv_TIO_L;
        fsSettings["TIO_R"] >> cv_TIO_R;
        cv::cv2eigen(cv_RIO, RIO);
        cv::cv2eigen(cv_TIO_L, TIO_L);
        cv::cv2eigen(cv_TIO_R, TIO_R);
        ROS_INFO_STREAM("RIO : " << std::endl << RIO);
        ROS_INFO_STREAM("TIO_L : " << std::endl << TIO_L.transpose());
        ROS_INFO_STREAM("TIO_R : " << std::endl << TIO_R.transpose());
    }

    fsSettings.release();
}
