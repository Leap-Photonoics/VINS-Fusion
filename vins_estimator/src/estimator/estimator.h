/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <atomic>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/gnss_vi_initializer.h"
#include "../factor/imu_factor.h"
#include "../factor/imu_encoder_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../factor/gnss_psr_dopp_factor.hpp"
#include "../factor/gnss_dt_ddt_factor.hpp"
#include "../factor/gnss_dt_anchor_factor.hpp"
#include "../factor/gnss_ddt_smooth_factor.hpp"
#include "../featureTracker/feature_tracker.h"
#include "segway_msgs/speed_fb.h"

#include <gnss_comm/gnss_utility.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_spp.hpp>

class Estimator
{
  public:
    Estimator();
    ~Estimator();
    void setParameter();

    template <typename T>
    struct time_pq_cmp
    {
        bool operator() (const pair<double, T> &a, const pair<double, T> &b) const 
        {
            return a.first > b.first;
        }
    };
    template <typename T> using time_pq = priority_queue<pair<double, shared_ptr<T>>, vector<pair<double, shared_ptr<T>>>, time_pq_cmp<shared_ptr<T>>>;

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const vector<shared_ptr<cv::Mat>> &_imgs);
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>> &image, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internalZ
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    void getIMUInterval(double t0, double t1, vector<pair<double, shared_ptr<Vector3d>>> &accVector, 
                                              vector<pair<double, shared_ptr<Vector3d>>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(int cam, set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Vector3d linear_acceleration, Vector3d angular_velocity);
    void initFirstIMUPose(vector<pair<double, shared_ptr<Vector3d>>> &accVector);

    // GNSS related
    bool GNSSVIAlign();
    void updateGNSSStatistics();
    void inputEphem(EphemBasePtr ephem_ptr);
    void inputIonoParams(double ts, const std::vector<double> &iono_params);
    void inputGNSSTimeDiff(const double t_diff);

    void inputGNSS(const double t, const std::vector<ObsPtr> &gnss_meas);
    void processGNSS(const shared_ptr<vector<ObsPtr>> &gnss_meas);
    void getGNSSInterval(double t0, double t1, vector<pair<double, shared_ptr<vector<ObsPtr>>>> &gnssVector);

    void inputEncoder(double t, double speed, double turn);
    void getEncoderInterval(double t0, double t1, vector<pair<double, shared_ptr<Matrix<double, 6, 1>>>> &encVector);
    void processIMUEncoder(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Matrix<double, 6, 1> &encoder_velocity);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;

    time_pq<Eigen::Vector3d> accBuf, gyrBuf;
    time_pq<Matrix<double,6,1>> encBuf;
    time_pq<map<int, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> featureBuf;
    time_pq<vector<ObsPtr>> gnssBuf;
    atomic<double> latest_imu_time, latest_encoder_time, latest_gnss_time;
    // queue<pair<double, Eigen::Vector3d>> accBuf;
    // queue<pair<double, Eigen::Vector3d>> gyrBuf;
    // queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    // queue<pair<double, vector<ObsPtr>>> gnssBuf;
    // deque<pair<double, Eigen::Vector3d>> encBuf;


    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    vector<shared_ptr<FeatureTracker>> featureTrackers;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[10];
    Vector3d tic[10];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;
    Matrix<double, 6, 1> enc_v_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
    vector<Matrix<double, 6, 1>> encoder_velocity_buf[(WINDOW_SIZE + 1)];

    // GNSS related
    bool gnss_ready;
    Eigen::Vector3d anc_ecef;
    Eigen::Matrix3d R_ecef_enu;
    double yaw_enu_local;
    std::vector<ObsPtr> gnss_meas_buf[(WINDOW_SIZE+1)];
    std::vector<EphemBasePtr> gnss_ephem_buf[(WINDOW_SIZE+1)];
    std::vector<double> latest_gnss_iono_params;
    std::map<uint32_t, std::vector<EphemBasePtr>> sat2ephem;
    std::map<uint32_t, std::map<double, size_t>> sat2time_index;
    std::map<uint32_t, uint32_t> sat_track_status;
    double para_anc_ecef[3];
    double para_yaw_enu_local[1];
    double para_rcv_dt[(WINDOW_SIZE+1)*4];
    double para_rcv_ddt[WINDOW_SIZE+1];
    // GNSS statistics
    double diff_t_gnss_local;
    Eigen::Matrix3d R_enu_local;
    Eigen::Vector3d ecef_pos, enu_pos, enu_vel, enu_ypr;

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    vector<shared_ptr<FeatureManager>> f_managers;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[10][NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[10][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    bool initThreadFlag;
};
