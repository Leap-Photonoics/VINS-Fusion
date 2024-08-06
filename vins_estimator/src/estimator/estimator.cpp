/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator()
{
    ROS_INFO("init begins");
    initThreadFlag = false;
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();
    while(!gnssBuf.empty())
        gnssBuf.pop();
    while(!encBuf.empty())
        encBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Vector3d(0, 0, 0);
    initR = Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        encoder_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    gnss_ready = false;
    anc_ecef.setZero();
    R_ecef_enu.setIdentity();
    para_yaw_enu_local[0] = 0;
    yaw_enu_local = 0;
    sat2ephem.clear();
    sat2time_index.clear();
    sat_track_status.clear();
    latest_gnss_iono_params.clear();
    std::copy(GNSS_IONO_DEFAULT_PARAMS.begin(), GNSS_IONO_DEFAULT_PARAMS.end(), 
        std::back_inserter(latest_gnss_iono_params));
    diff_t_gnss_local = 0;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    for (auto f_manager: f_managers)
        f_manager->clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
    {
        featureTrackers.push_back(make_shared<FeatureTracker>());
        f_managers.push_back(make_shared<FeatureManager>(Rs, cam));
    }
    clearState();

    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
        featureTrackers[cam]->readIntrinsicParameter(vector<string>{CAM_NAMES.begin() + cam * 2, CAM_NAMES.begin() + cam * 2 + 2});

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const vector<shared_ptr<cv::Mat>> &_imgs)
{
    inputImageCnt++;
    map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>> featureFrame;
    TicToc featureTrackerTime;

    if(NUM_OF_CAM == 1)
        featureFrame[0] = featureTrackers[0]->trackImage(t, _imgs[0]);
    else
        cv::parallel_for_(cv::Range(0, NUM_OF_CAM_UNIT), [&](const cv::Range &range)
        {
            for (int cam = range.start; cam < range.end; ++cam)
                featureFrame[cam] = featureTrackers[cam]->trackImage(t, _imgs[cam * 2], _imgs[cam * 2 + 1]);
        });
    // printf("featureTracker time: %f\n", featureTrackerTime.toc());

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTrackers[0]->getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    
    if(MULTIPLE_THREAD)  
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.emplace(t, make_shared<map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>>>(featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.emplace(t, make_shared<map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>>>(featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    latest_imu_time = t;
    accBuf.emplace(t, make_shared<Vector3d>(linearAcceleration));
    gyrBuf.emplace(t, make_shared<Vector3d>(angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputEncoder(double t, double speed_l, double speed_r)
{
    std::lock_guard<std::mutex> lg(mBuf);
    latest_encoder_time = t;
    Matrix<double, 6, 1> vel;
    vel << 0, 0, speed_l, 0, 0, speed_r;
    encBuf.emplace(t, make_shared<Matrix<double, 6, 1>>(vel));
}

void Estimator::getEncoderInterval(double t0, double t1, vector<pair<double, shared_ptr<Matrix<double, 6, 1>>>> &encVector)
{
    std::lock_guard<std::mutex> lg(mBuf);
    while (encBuf.top().first < t1)
    {
        encVector.push_back(encBuf.top());
        encBuf.pop();
    }
    encVector.push_back(encBuf.top());
    if (encVector.size() > 1)
        encBuf.push(encVector[encVector.size() - 2]);
}

void Estimator::getIMUInterval(double t0, double t1, vector<pair<double, shared_ptr<Vector3d>>> &accVector, 
                                vector<pair<double, shared_ptr<Vector3d>>> &gyrVector)
{
    while (accBuf.top().first <= t0)
    {
        accBuf.pop();
        gyrBuf.pop();
    }
    while (accBuf.top().first < t1)
    {
        accVector.push_back(accBuf.top());
        accBuf.pop();
        gyrVector.push_back(gyrBuf.top());
        gyrBuf.pop();
    }
    accVector.push_back(accBuf.top());
    gyrVector.push_back(gyrBuf.top());
}

void Estimator::getGNSSInterval(double t0, double t1, vector<pair<double, shared_ptr<vector<ObsPtr>>>> &gnssVector)
{
    std::lock_guard<std::mutex> lg(mBuf);
    while (gnssBuf.top().first <= t0)
        gnssBuf.pop();
    while (gnssBuf.top().first < t1)
    {
        gnssVector.push_back(gnssBuf.top());
        gnssBuf.pop();
    }
}

void Estimator::inputEphem(EphemBasePtr ephem_ptr)
{
    double toe = time2sec(ephem_ptr->toe);
    // if a new ephemeris comes
    if (sat2time_index.count(ephem_ptr->sat) == 0 || sat2time_index.at(ephem_ptr->sat).count(toe) == 0)
    {
        sat2ephem[ephem_ptr->sat].emplace_back(ephem_ptr);
        sat2time_index[ephem_ptr->sat].emplace(toe, sat2ephem.at(ephem_ptr->sat).size()-1);
    }
}

void Estimator::inputIonoParams(double ts, const std::vector<double> &iono_params)
{
    if (iono_params.size() != 8)    return;

    // update ionosphere parameters
    latest_gnss_iono_params.clear();
    std::copy(iono_params.begin(), iono_params.end(), std::back_inserter(latest_gnss_iono_params));
}

void Estimator::inputGNSSTimeDiff(const double t_diff)
{
    diff_t_gnss_local = t_diff;
}

void Estimator::inputGNSS(const double t, const vector<ObsPtr> &gnss_meas)
{
    std::lock_guard<std::mutex> lg(mBuf);
    latest_gnss_time = t;
    gnssBuf.emplace(t, make_shared<vector<ObsPtr>>(gnss_meas));
}

void Estimator::processGNSS(const shared_ptr<vector<ObsPtr>> &gnss_meas)
{
    std::vector<ObsPtr> valid_meas;
    std::vector<EphemBasePtr> valid_ephems;
    for (auto obs : *gnss_meas)
    {
        // filter according to system
        uint32_t sys = satsys(obs->sat, NULL);
        if (sys != SYS_GPS && sys != SYS_GLO && sys != SYS_GAL && sys != SYS_BDS)
            continue;

        // if not got cooresponding ephemeris yet
        if (sat2ephem.count(obs->sat) == 0)
            continue;
        
        if (obs->freqs.empty())    continue;       // no valid signal measurement
        int freq_idx = -1;
        L1_freq(obs, &freq_idx);
        if (freq_idx < 0)   continue;              // no L1 observation
        
        double obs_time = time2sec(obs->time);
        std::map<double, size_t> time2index = sat2time_index.at(obs->sat);
        double ephem_time = EPH_VALID_SECONDS;
        size_t ephem_index = -1;
        for (auto ti : time2index)
        {
            if (std::abs(ti.first - obs_time) < ephem_time)
            {
                ephem_time = std::abs(ti.first - obs_time);
                ephem_index = ti.second;
            }
        }
        if (ephem_time >= EPH_VALID_SECONDS)
        {
            cerr << "ephemeris not valid anymore\n";
            continue;
        }
        const EphemBasePtr &best_ephem = sat2ephem.at(obs->sat).at(ephem_index);

        // filter by tracking status
        LOG_IF(FATAL, freq_idx < 0) << "No L1 observation found.\n";
        if (obs->psr_std[freq_idx]  > GNSS_PSR_STD_THRES ||
            obs->dopp_std[freq_idx] > GNSS_DOPP_STD_THRES)
        {
            sat_track_status[obs->sat] = 0;
            continue;
        }
        else
        {
            if (sat_track_status.count(obs->sat) == 0)
                sat_track_status[obs->sat] = 0;
            ++ sat_track_status[obs->sat];
        }
        if (sat_track_status[obs->sat] < GNSS_TRACK_NUM_THRES)
            continue;           // not being tracked for enough epochs

        // filter by elevation angle
        if (gnss_ready)
        {
            Vector3d sat_ecef;
            if (sys == SYS_GLO)
                sat_ecef = geph2pos(obs->time, std::dynamic_pointer_cast<GloEphem>(best_ephem), NULL);
            else
                sat_ecef = eph2pos(obs->time, std::dynamic_pointer_cast<Ephem>(best_ephem), NULL);
            double azel[2] = {0, M_PI/2.0};
            sat_azel(ecef_pos, sat_ecef, azel);
            if (azel[1] < GNSS_ELEVATION_THRES*M_PI/180.0)
                continue;
        }
        valid_meas.push_back(obs);
        valid_ephems.push_back(best_ephem);
    }
    
    gnss_meas_buf[frame_count] = valid_meas;
    gnss_ephem_buf[frame_count] = valid_ephems;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, shared_ptr<map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>>>> feature;
        vector<pair<double, shared_ptr<Vector3d>>> accVector, gyrVector;
        if (!featureBuf.empty())
        {
            feature = featureBuf.top();
            curTime = feature.first + td;
            bool imu_late_msg_printed = false;
            while (USE_IMU && latest_imu_time < curTime)
            {
                if (!imu_late_msg_printed)
                {
                    imu_late_msg_printed = true;
                    printf("wait for imu ... \n");
                }
                if (! MULTIPLE_THREAD)
                    return;
                std::chrono::milliseconds dura(5);
                std::this_thread::sleep_for(dura);
            }
            mBuf.lock();
            if (USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            mBuf.unlock();

            vector<pair<double, shared_ptr<vector<ObsPtr>>>> gnssVector;
            vector<pair<double, shared_ptr<Matrix<double, 6, 1>>>> encVector;

            if (USE_IMU)
            {
                if (!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                if (ENCODER_ENABLE)
                {
                    bool enc_late_msg_printed = false;
                    while (latest_encoder_time < curTime)
                    {
                        if (!enc_late_msg_printed)
                        {
                            enc_late_msg_printed = true;
                            printf("wait for encoder ... \n");
                        }
                        std::chrono::milliseconds dura(5);
                        std::this_thread::sleep_for(dura);
                    }
                    getEncoderInterval(prevTime, curTime, encVector);
                    Matrix<double, 6, 1> last_velocity;
                    for(size_t i = 0; i < accVector.size(); i++)
                    {
                        double t = accVector[i].first, t0 = 0, t1 = 0;
                        Matrix<double, 6, 1> encoder_velocity;
                        if (!encVector.empty())
                            encoder_velocity = *(encVector[0].second);
                        else
                        {
                            encoder_velocity.block<3, 1>(0, 0) = Vs[frame_count];
                            encoder_velocity.block<3, 1>(3, 0) = Vs[frame_count];
                        }
                        Matrix<double, 6, 1> vel0, vel1;
                        for (auto &enc_vel : encVector)
                        {
                            if (enc_vel.first <= t)
                            {
                                t0 = enc_vel.first;
                                vel0 = *(enc_vel.second);
                            }
                            else
                            {
                                t1 = enc_vel.first;
                                vel1 = *(enc_vel.second);
                                break;
                            }
                        }
                        if (t0 > 0 && t1 > 0)
                        {
                            double dt0 = t - t0, dt1 = t1 - t;
                            ROS_ASSERT(dt0 >= 0);
                            ROS_ASSERT(dt1 >= 0);
                            ROS_ASSERT(dt0 + dt1 > 0);
                            double w1 = dt1 / (dt0 + dt1), w2 = dt0 / (dt0 + dt1);
                            encoder_velocity = w1 * vel0 + w2 * vel1;
                        }
                        double dt;
                        if(i == 0)
                            dt = accVector[i].first - prevTime;
                        else if (i == accVector.size() - 1)
                            dt = curTime - accVector[i - 1].first;
                        else
                            dt = accVector[i].first - accVector[i - 1].first;
                        ROS_ASSERT(dt >= 0);
                        if (t <= curTime)
                            processIMUEncoder(dt, *(accVector[i].second), *(gyrVector[i].second), encoder_velocity);
                        else 
                        {
                            double dt1 = dt, dt2 = t - curTime;
                            ROS_ASSERT(dt1 >= 0);
                            ROS_ASSERT(dt2 >= 0);
                            ROS_ASSERT(dt1 + dt2 > 0);
                            double w1 = dt2 / (dt1 + dt2), w2 = dt1 / (dt1 + dt2);
                            processIMUEncoder(dt, w1 * *(accVector[i-1].second) + w2 * *(accVector[i].second), 
                                                  w1 * *(gyrVector[i-1].second) + w2 * *(gyrVector[i].second), 
                                                  w1 * last_velocity + w2 * encoder_velocity);
                        }
                        last_velocity = encoder_velocity;
                    }
                }
                else for(size_t i = 0; i < accVector.size(); i++)
                    {
                        double dt;
                        if(i == 0)
                            dt = accVector[i].first - prevTime;
                        else if (i == accVector.size() - 1)
                            dt = curTime - accVector[i - 1].first;
                        else
                            dt = accVector[i].first - accVector[i - 1].first;
                        processIMU(accVector[i].first, dt, *(accVector[i].second), *(gyrVector[i].second));
                    }
            }

            if (GNSS_ENABLE)
            {
                bool gnss_late_msg_printed = false;
                while (latest_gnss_time < curTime)
                {
                    if (!gnss_late_msg_printed)
                    {
                        gnss_late_msg_printed = true;
                        printf("wait for gnss ... \n");
                    }
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
                getGNSSInterval(prevTime, curTime, gnssVector);
                for (auto gnssMeas: gnssVector)
                    processGNSS(gnssMeas.second);
            }
            

            mProcess.lock();
            processImage(*(feature.second), feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, shared_ptr<Vector3d>>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Vector3d averAcc(0, 0, 0);
    int n = accVector.size();
    for(size_t i = 0; i < n; i++)
    {
        averAcc = averAcc + *(accVector[i].second);
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Vector3d p, Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const map<int, map<int, vector<pair<int, Matrix<double, 7, 1>>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    marginalization_flag = MARGIN_OLD;
    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
        if (!f_managers[cam]->addFeatureCheckParallax(frame_count, image.at(cam), td))
            marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_managers[0]->getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    if (ENCODER_ENABLE)
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], enc_v_0};
    else
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            for (auto f_manager : f_managers)
            {
                f_manager->initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
                f_manager->triangulate(frame_count, Ps, Rs, tic, ric);
            }
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            for (auto f_manager : f_managers)
            {
                f_manager->initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
                f_manager->triangulate(frame_count, Ps, Rs, tic, ric);
            }
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        for (auto f_manager : f_managers)
        {
            if(!USE_IMU)
                f_manager->initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager->triangulate(frame_count, Ps, Rs, tic, ric);
        }
        optimization();
        if (GNSS_ENABLE)
        {
            if (!gnss_ready)
                gnss_ready = GNSSVIAlign();
            if (gnss_ready)
                updateGNSSStatistics();
        }
        for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
        {
            set<int> removeIndex;
            outliersRejection(cam, removeIndex);
            f_managers[cam]->removeOutlier(removeIndex);
            if (! MULTIPLE_THREAD)
            {
                featureTrackers[cam]->removeOutliers(removeIndex);
                predictPtsInNextFrame();
            }
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        for (auto f_manager: f_managers)
            f_manager->removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}

void Estimator::processIMUEncoder(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Matrix<double, 6, 1> &encoder_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        enc_v_0 = encoder_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count], enc_v_0};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity, encoder_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity, encoder_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        encoder_velocity_buf[frame_count].push_back(encoder_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
    enc_v_0 = encoder_velocity;
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_managers[0]->feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points.at(0))
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    for (auto f_manager : f_managers)
    {
        f_manager->clearDepth();
        f_manager->triangulate(frame_count, Ps, Rs, tic, ric);
    }

    return true;
}

bool Estimator::GNSSVIAlign()
{
    if (solver_flag == INITIAL)     // visual-inertial not initialized
        return false;
    
    if (gnss_ready)                 // GNSS-VI already initialized
        return true;
    
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
    {
        if (gnss_meas_buf[i].empty() || gnss_meas_buf[i].size() < 10) 
            return false;
    }

    // check horizontal velocity excitation
    Vector2d avg_hor_vel(0.0, 0.0);
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
        avg_hor_vel += Vs[i].head<2>().cwiseAbs();
    avg_hor_vel /= (WINDOW_SIZE+1);
    if (avg_hor_vel.norm() < 0.3)
    {
        std::cerr << "velocity excitation not enough for GNSS-VI alignment.\n";
        return false;
    }

    std::vector<std::vector<ObsPtr>> curr_gnss_meas_buf;
    std::vector<std::vector<EphemBasePtr>> curr_gnss_ephem_buf;
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
    {
        curr_gnss_meas_buf.push_back(gnss_meas_buf[i]);
        curr_gnss_ephem_buf.push_back(gnss_ephem_buf[i]);
    }

    GNSSVIInitializer gnss_vi_initializer(curr_gnss_meas_buf, curr_gnss_ephem_buf, latest_gnss_iono_params);

    // 1. get a rough global location
    Matrix<double, 7, 1> rough_xyzt;
    rough_xyzt.setZero();
    if (!gnss_vi_initializer.coarse_localization(rough_xyzt))
    {
        std::cerr << "Fail to obtain a coarse location.\n";
        return false;
    }

    // 2. perform yaw alignment
    std::vector<Vector3d> local_vs;
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
        local_vs.push_back(Vs[i]);
    Vector3d rough_anchor_ecef = rough_xyzt.head<3>();
    double aligned_yaw = 0;
    double aligned_rcv_ddt = 0;
    if (!gnss_vi_initializer.yaw_alignment(local_vs, rough_anchor_ecef, aligned_yaw, aligned_rcv_ddt))
    {
        std::cerr << "Fail to align ENU and local frames.\n";
        return false;
    }
    // std::cout << "aligned_yaw is " << aligned_yaw*180.0/M_PI << '\n';

    // 3. perform anchor refinement
    std::vector<Vector3d> local_ps;
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
        local_ps.push_back(Ps[i]);
    Matrix<double, 7, 1> refined_xyzt;
    refined_xyzt.setZero();
    if (!gnss_vi_initializer.anchor_refinement(local_ps, aligned_yaw, 
        aligned_rcv_ddt, rough_xyzt, refined_xyzt))
    {
        std::cerr << "Fail to refine anchor point.\n";
        return false;
    }
    // std::cout << "refined anchor point is " << std::setprecision(20) 
    //           << refined_xyzt.head<3>().transpose() << '\n';

    // restore GNSS states
    uint32_t one_observed_sys = static_cast<uint32_t>(-1);
    for (uint32_t k = 0; k < 4; ++k)
    {
        if (rough_xyzt(k+3) != 0)
        {
            one_observed_sys = k;
            break;
        }
    }
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
    {
        para_rcv_ddt[i] = aligned_rcv_ddt;
        for (uint32_t k = 0; k < 4; ++k)
        {
            if (rough_xyzt(k+3) == 0)
                para_rcv_dt[i*4+k] = refined_xyzt(3+one_observed_sys) + aligned_rcv_ddt * i;
            else
                para_rcv_dt[i*4+k] = refined_xyzt(3+k) + aligned_rcv_ddt * i;
        }
    }
    anc_ecef = refined_xyzt.head<3>();
    R_ecef_enu = ecef2rotation(anc_ecef);

    yaw_enu_local = aligned_yaw;

    return true;
}

void Estimator::updateGNSSStatistics()
{
    R_enu_local = AngleAxisd(yaw_enu_local, Vector3d::UnitZ());
    enu_pos = R_enu_local * Ps[WINDOW_SIZE];
    enu_vel = R_enu_local * Vs[WINDOW_SIZE];
    enu_ypr = Utility::R2ypr(R_enu_local*Rs[WINDOW_SIZE]);
    ecef_pos = anc_ecef + R_ecef_enu * enu_pos;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_managers[0]->getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
    {
        VectorXd dep = f_managers[cam]->getDepthVector();
        for (int i = 0; i < f_managers[cam]->getFeatureCount(); ++i)
            para_Feature[cam][i][0] = dep(i);
    }

    para_Td[0][0] = td;
    
    para_yaw_enu_local[0] = yaw_enu_local;
    for (uint32_t k = 0; k < 3; ++k)
        para_anc_ecef[k] = anc_ecef(k);
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                                para_SpeedBias[i][4],
                                para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                para_SpeedBias[i][7],
                                para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
    {
        VectorXd dep = f_managers[cam]->getDepthVector();
        for (int i = 0; i < f_managers[cam]->getFeatureCount(); i++)
            dep(i) = para_Feature[cam][i][0];
        f_managers[cam]->setDepth(dep);
    }

    if(USE_IMU)
        td = para_Td[0][0];
    
    if (gnss_ready)
    {
        yaw_enu_local = para_yaw_enu_local[0];
        for (uint32_t k = 0; k < 3; ++k)
            anc_ecef(k) = para_anc_ecef[k];
        R_ecef_enu = ecef2rotation(anc_ecef);
    }

}

bool Estimator::failureDetection()
{
    return false;
    for (auto f_manager: f_managers)
        if (f_manager->last_track_num < 2)
        {
            ROS_INFO(" little feature %d", f_manager->last_track_num);
            //return true;
        }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (gnss_ready)
    {
        problem.AddParameterBlock(para_yaw_enu_local, 1);
        Vector2d avg_hor_vel(0.0, 0.0);
        for (uint32_t i = 0; i <= WINDOW_SIZE; ++i)
            avg_hor_vel += Vs[i].head<2>().cwiseAbs();
        avg_hor_vel /= (WINDOW_SIZE+1);
        // cerr << "avg_hor_vel is " << avg_vel << endl;
        if (avg_hor_vel.norm() < 0.3)
        {
            // std::cerr << "velocity excitation not enough, fix yaw angle.\n";
            problem.SetParameterBlockConstant(para_yaw_enu_local);
        }

        for (uint32_t i = 0; i <= WINDOW_SIZE; ++i)
        {
            if (gnss_meas_buf[i].size() < 10)
                problem.SetParameterBlockConstant(para_yaw_enu_local);
        }
        
        problem.AddParameterBlock(para_anc_ecef, 3);
        // problem.SetParameterBlockConstant(para_anc_ecef);

        for (uint32_t i = 0; i <= WINDOW_SIZE; ++i)
        {
            for (uint32_t k = 0; k < 4; ++k)
                problem.AddParameterBlock(para_rcv_dt+i*4+k, 1);
            problem.AddParameterBlock(para_rcv_ddt+i, 1);
        }
    }

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            if (ENCODER_ENABLE)
            {
                IMUEncoderFactor* imu_encoder_factor = new IMUEncoderFactor(pre_integrations[j]);
                problem.AddResidualBlock(imu_encoder_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
            } 
            else 
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
                problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
            }
        }
    }

    if (gnss_ready)
    {
        for(int i = 0; i <= WINDOW_SIZE; ++i)
        {
            // cerr << "size of gnss_meas_buf[" << i << "] is " << gnss_meas_buf[i].size() << endl;
            const std::vector<ObsPtr> &curr_obs = gnss_meas_buf[i];
            const std::vector<EphemBasePtr> &curr_ephem = gnss_ephem_buf[i];

            for (uint32_t j = 0; j < curr_obs.size(); ++j)
            {
                const uint32_t sys = satsys(curr_obs[j]->sat, NULL);
                const uint32_t sys_idx = gnss_comm::sys2idx.at(sys);

                int lower_idx = -1;
                const double obs_local_ts = time2sec(curr_obs[j]->time) - diff_t_gnss_local;
                if (Headers[i] > obs_local_ts)
                    lower_idx = (i==0? 0 : i-1);
                else
                    lower_idx = (i==WINDOW_SIZE? WINDOW_SIZE-1 : i);
                const double lower_ts = Headers[lower_idx];
                const double upper_ts = Headers[lower_idx+1];

                const double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);
                GnssPsrDoppFactor *gnss_factor = new GnssPsrDoppFactor(curr_obs[j], 
                    curr_ephem[j], latest_gnss_iono_params, ts_ratio);
                problem.AddResidualBlock(gnss_factor, NULL, para_Pose[lower_idx], 
                    para_SpeedBias[lower_idx], para_Pose[lower_idx+1], para_SpeedBias[lower_idx+1],
                    para_rcv_dt+i*4+sys_idx, para_rcv_ddt+i, para_yaw_enu_local, para_anc_ecef);
            }
        }

        // build relationship between rcv_dt and rcv_ddt
        for (size_t k = 0; k < 4; ++k)
        {
            for (uint32_t i = 0; i < WINDOW_SIZE; ++i)
            {
                const double gnss_dt = Headers[i+1] - Headers[i];
                DtDdtFactor *dt_ddt_factor = new DtDdtFactor(gnss_dt);
                problem.AddResidualBlock(dt_ddt_factor, NULL, para_rcv_dt+i*4+k, 
                    para_rcv_dt+(i+1)*4+k, para_rcv_ddt+i, para_rcv_ddt+i+1);
            }
        }

        // add rcv_ddt smooth factor
        for (int i = 0; i < WINDOW_SIZE; ++i)
        {
            DdtSmoothFactor *ddt_smooth_factor = new DdtSmoothFactor(GNSS_DDT_WEIGHT);
            problem.AddResidualBlock(ddt_smooth_factor, NULL, para_rcv_ddt+i, para_rcv_ddt+i+1);
        }
    }

    int f_m_cnt = 0;
    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
    {
        int feature_index = -1;
        for (auto &it_per_id : f_managers[cam]->feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
    
            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[cam * 2], para_Feature[cam][feature_index], para_Td[0]);
                }

                if(STEREO && it_per_frame.is_stereo)
                {                
                    Vector3d pts_j_right = it_per_frame.pointRight;
                    if(imu_i != imu_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[cam * 2], para_Ex_Pose[cam * 2 + 1], para_Feature[cam][feature_index], para_Td[0]);
                    }
                    else
                    {
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        problem.AddResidualBlock(f, loss_function, para_Ex_Pose[cam * 2], para_Ex_Pose[cam * 2 + 1], para_Feature[cam][feature_index], para_Td[0]);
                    }
                
                }
                f_m_cnt++;
            }
        }
    }
    
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    while(para_yaw_enu_local[0] > M_PI)   para_yaw_enu_local[0] -= 2.0*M_PI;
    while(para_yaw_enu_local[0] < -M_PI)  para_yaw_enu_local[0] += 2.0*M_PI;

    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                if (ENCODER_ENABLE)
                {
                    IMUEncoderFactor* imu_encoder_factor = new IMUEncoderFactor(pre_integrations[1]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_encoder_factor, NULL, 
                                                                            vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}, 
                                                                            vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                else
                {
                    IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                            vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                            vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        if (gnss_ready)
        {
            for (uint32_t j = 0; j < gnss_meas_buf[0].size(); ++j)
            {
                const uint32_t sys = satsys(gnss_meas_buf[0][j]->sat, NULL);
                const uint32_t sys_idx = gnss_comm::sys2idx.at(sys);

                const double obs_local_ts = time2sec(gnss_meas_buf[0][j]->time) - diff_t_gnss_local;
                const double lower_ts = Headers[0];
                const double upper_ts = Headers[1];
                const double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);

                GnssPsrDoppFactor *gnss_factor = new GnssPsrDoppFactor(gnss_meas_buf[0][j], 
                    gnss_ephem_buf[0][j], latest_gnss_iono_params, ts_ratio);
                ResidualBlockInfo *psr_dopp_residual_block_info = new ResidualBlockInfo(gnss_factor, NULL,
                    vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], 
                        para_SpeedBias[1],para_rcv_dt+sys_idx, para_rcv_ddt, 
                        para_yaw_enu_local, para_anc_ecef},
                    vector<int>{0, 1, 4, 5});
                marginalization_info->addResidualBlockInfo(psr_dopp_residual_block_info);
            }

            const double gnss_dt = Headers[1] - Headers[0];
            for (size_t k = 0; k < 4; ++k)
            {
                DtDdtFactor *dt_ddt_factor = new DtDdtFactor(gnss_dt);
                ResidualBlockInfo *dt_ddt_residual_block_info = new ResidualBlockInfo(dt_ddt_factor, NULL,
                    vector<double *>{para_rcv_dt+k, para_rcv_dt+4+k, para_rcv_ddt, para_rcv_ddt+1}, 
                    vector<int>{0, 2});
                marginalization_info->addResidualBlockInfo(dt_ddt_residual_block_info);
            }

            // margin rcv_ddt smooth factor
            DdtSmoothFactor *ddt_smooth_factor = new DdtSmoothFactor(GNSS_DDT_WEIGHT);
            ResidualBlockInfo *ddt_smooth_residual_block_info = new ResidualBlockInfo(ddt_smooth_factor, NULL,
                    vector<double *>{para_rcv_ddt, para_rcv_ddt+1}, vector<int>{0});
            marginalization_info->addResidualBlockInfo(ddt_smooth_residual_block_info);
        }

        for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
        {
            int feature_index = -1;
            for (auto &it_per_id : f_managers[cam]->feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[cam * 2], para_Feature[cam][feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[cam * 2], para_Ex_Pose[cam * 2 + 1], para_Feature[cam][feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[cam * 2], para_Ex_Pose[cam * 2 + 1], para_Feature[cam][feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            for (uint32_t k = 0; k < 4; ++k)
                addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+(i-1)*4+k;
            addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i-1;
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        addr_shift[reinterpret_cast<long>(para_yaw_enu_local)] = para_yaw_enu_local;
        addr_shift[reinterpret_cast<long>(para_anc_ecef)] = para_anc_ecef;

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                    for (uint32_t k = 0; k < 4; ++k)
                        addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+(i-1)*4+k;
                    addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i-1;
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                    for (uint32_t k = 0; k < 4; ++k)
                        addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+i*4+k;
                    addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i;
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            addr_shift[reinterpret_cast<long>(para_yaw_enu_local)] = para_yaw_enu_local;
            addr_shift[reinterpret_cast<long>(para_anc_ecef)] = para_anc_ecef;

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
                    encoder_velocity_buf[i].swap(encoder_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }

                // GNSS related
                gnss_meas_buf[i].swap(gnss_meas_buf[i+1]);
                gnss_ephem_buf[i].swap(gnss_ephem_buf[i+1]);
                for (uint32_t k = 0; k < 4; ++k)
                    para_rcv_dt[i*4+k] = para_rcv_dt[(i+1)*4+k];
                para_rcv_ddt[i] = para_rcv_ddt[i+1];
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            // GNSS related
            gnss_meas_buf[WINDOW_SIZE].clear();
            gnss_ephem_buf[WINDOW_SIZE].clear();

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                if (ENCODER_ENABLE)
                    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE], enc_v_0};
                else
                    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
                encoder_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    if (ENCODER_ENABLE) {
                        Matrix<double, 6, 1> tmp_encoder_velocity = encoder_velocity_buf[frame_count][i];
                        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity, tmp_encoder_velocity);
                        encoder_velocity_buf[frame_count - 1].push_back(tmp_encoder_velocity);
                    }
                    else
                        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                // GNSS related
                gnss_meas_buf[frame_count-1] = gnss_meas_buf[frame_count];
                gnss_ephem_buf[frame_count-1] = gnss_ephem_buf[frame_count];
                for (uint32_t k = 0; k < 4; ++k)
                    para_rcv_dt[(frame_count-1)*4+k] = para_rcv_dt[frame_count*4+k];
                para_rcv_ddt[frame_count-1] = para_rcv_ddt[frame_count];
                gnss_meas_buf[frame_count].clear();
                gnss_ephem_buf[frame_count].clear();

                delete pre_integrations[WINDOW_SIZE];
                if (ENCODER_ENABLE)
                    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE], enc_v_0};
                else
                    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    for (auto f_manager: f_managers)
        f_manager->removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
        for (int cam  = 0; cam < NUM_OF_CAM_UNIT; ++ cam)
        {
            Matrix3d R0, R1;
            Vector3d P0, P1;
            R0 = back_R0 * ric[cam * 2];
            R1 = Rs[0] * ric[cam * 2];
            P0 = back_P0 + back_R0 * tic[cam * 2];
            P1 = Ps[0] + Rs[0] * tic[cam * 2];
            f_managers[cam]->removeBackShiftDepth(R0, P0, R1, P1);
        }
    else
        for (auto f_manager: f_managers)
            f_manager->removeBack();
}


void Estimator::getPoseInWorldFrame(Matrix4d &T)
{
    T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Matrix4d &T)
{
    T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    
    for (int cam = 0; cam < NUM_OF_CAM_UNIT; ++cam)
    {
        map<int, Vector3d> predictPts;
        for (auto &it_per_id : f_managers[cam]->feature)
        {
            if(it_per_id.estimated_depth > 0)
            {
                int firstIndex = it_per_id.start_frame;
                int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
                //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
                if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
                {
                    double depth = it_per_id.estimated_depth;
                    Vector3d pts_j = ric[cam * 2] * (depth * it_per_id.feature_per_frame[0].point) + tic[cam * 2];
                    Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                    Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                    Vector3d pts_cam = ric[cam * 2].transpose() * (pts_local - tic[cam * 2]);
                    int ptsIndex = it_per_id.feature_id;
                    predictPts[ptsIndex] = pts_cam;
                }
            }
        }
        featureTrackers[cam]->setPrediction(predictPts);
        //printf("estimator output %d predict pts\n",(int)predictPts.size());
    }
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(int cam, set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_managers[cam]->feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[cam * 2], tic[cam * 2], 
                                                    Rs[imu_j], Ps[imu_j], ric[cam * 2], tic[cam * 2],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[cam * 2], tic[cam * 2], 
                                                        Rs[imu_j], Ps[imu_j], ric[cam * 2 + 1], tic[cam * 2 + 1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[cam * 2], tic[cam * 2], 
                                                        Rs[imu_j], Ps[imu_j], ric[cam * 2 + 1], tic[cam * 2 + 1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Vector3d linear_acceleration, Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    time_pq<Vector3d> tmp_accBuf(accBuf), tmp_gyrBuf(gyrBuf);
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        fastPredictIMU(tmp_accBuf.top().first, *(tmp_accBuf.top().second), *(tmp_gyrBuf.top().second));
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
