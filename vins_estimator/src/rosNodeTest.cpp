/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

double next_pulse_time;
bool next_pulse_time_valid;
double time_diff_gnss_local;
bool time_diff_valid;

double encoder_time_diff;
bool encoder_time_diff_valid = false;

vector<queue<sensor_msgs::ImageConstPtr>> img_buffer;
std::mutex m_buf;

shared_ptr<cv::Mat> getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    return make_shared<cv::Mat>(ptr->image);
}

void img_callback(const int cam_id, const sensor_msgs::ImageConstPtr& img) 
{
    lock_guard<mutex> lock(m_buf);
    img_buffer[cam_id].push(img);
}

void stereo_sync()
{
    static const double max_dt = 0.01;
    while (1)
    {
        unique_lock<mutex> m_buf_lock(m_buf);
        bool buffer_empty = false;
        vector<shared_ptr<cv::Mat>> imgs;
        double t;
        for (int i = 0; i < NUM_OF_CAM; i++)
            if (img_buffer[i].empty())
            {
                buffer_empty = true;
                break;
            }
        if (!buffer_empty)
        {
            bool all_sync = true;
            for (int i = 1; i < NUM_OF_CAM; i++)
            {
                double dt = img_buffer[i].front()->header.stamp.toSec() - img_buffer[0].front()->header.stamp.toSec();    
                if (dt > max_dt)
                {
                    img_buffer[0].pop();
                    ROS_INFO("throw image 0");
                    all_sync = false;
                    break;
                }
                if (dt < -max_dt)
                {
                    img_buffer[i].pop();
                    ROS_INFO("throw image %d", i);
                    all_sync = false;
                    break;
                }
            }
            if (all_sync)
            {
                t = img_buffer[0].front()->header.stamp.toSec();
                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    imgs.push_back(getImageFromMsg(img_buffer[i].front()));
                    img_buffer[i].pop();
                }
            }
        }
        m_buf_lock.unlock();
        if (!imgs.empty())
            estimator.inputImage(t, imgs);

        std::chrono::milliseconds dura(30);
        std::this_thread::sleep_for(dura);
    }
}

void mono_callback(const sensor_msgs::ImageConstPtr& img0) 
{
    estimator.inputImage(img0->header.stamp.toSec(), {getImageFromMsg(img0)});
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void gnss_ephem_callback(const GnssEphemMsgConstPtr &ephem_msg)
{
    EphemPtr ephem = msg2ephem(ephem_msg);
    estimator.inputEphem(ephem);
}

void gnss_glo_ephem_callback(const GnssGloEphemMsgConstPtr &glo_ephem_msg)
{
    GloEphemPtr glo_ephem = msg2glo_ephem(glo_ephem_msg);
    estimator.inputEphem(glo_ephem);
}

void gnss_iono_params_callback(const StampedFloat64ArrayConstPtr &iono_msg)
{
    double ts = iono_msg->header.stamp.toSec();
    std::vector<double> iono_params;
    std::copy(iono_msg->data.begin(), iono_msg->data.end(), std::back_inserter(iono_params));
    assert(iono_params.size() == 8);
    estimator.inputIonoParams(ts, iono_params);
}

void gnss_meas_callback(const GnssMeasMsgConstPtr &meas_msg)
{
    std::vector<ObsPtr> gnss_meas = msg2meas(meas_msg);

    // cerr << "gnss ts is " << std::setprecision(20) << time2sec(gnss_meas[0]->time) << endl;
    if (!time_diff_valid)   return;

    estimator.inputGNSS(time2sec(gnss_meas[0]->time) - time_diff_gnss_local, gnss_meas);
}

void local_trigger_info_callback(const sensor_msgs::ImageConstPtr &msg)
{
    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - msg->header.stamp.toSec();
        estimator.inputGNSSTimeDiff(time_diff_gnss_local);
        if (!time_diff_valid)       // just get calibrated
            std::cout << "time difference between GNSS and VI-Sensor got calibrated: "
                << std::setprecision(15) << time_diff_gnss_local << " s\n";
        time_diff_valid = true;
    }
}

void gnss_tp_info_callback(const GnssTimePulseInfoMsgConstPtr &tp_msg)
{
    gtime_t tp_time = gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = time2sec(tp_time);

    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void encoder_callback(const segway_msgs::speed_fbConstPtr &msg)
{
    if (!encoder_time_diff_valid)
    {
        encoder_time_diff = msg->header.stamp.toSec() - msg->speed_timestamp * 1e-6;
        encoder_time_diff_valid = true;
    }
    estimator.inputEncoder(msg->speed_timestamp * 1e-6 + encoder_time_diff, msg->rl_speed,  msg->rr_speed);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    next_pulse_time_valid = false;
    time_diff_valid = false;

    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }

    vector<shared_ptr<ros::Subscriber>> sub_imgs;

    ROS_ASSERT(NUM_OF_CAM == 1 || NUM_OF_CAM == 2 || NUM_OF_CAM == 4);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        sub_imgs.push_back(make_shared<ros::Subscriber>(n.subscribe<sensor_msgs::Image>(IMAGE_TOPICS[i], 100, bind(img_callback, i, _1))));
        img_buffer.push_back(queue<sensor_msgs::ImageConstPtr>());
    }

    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);

    ros::Subscriber sub_ephem, sub_glo_ephem, sub_gnss_meas, sub_gnss_iono_params;
    ros::Subscriber sub_gnss_time_pluse_info, sub_local_trigger_info;
    if (GNSS_ENABLE)
    {
        sub_ephem = n.subscribe(GNSS_EPHEM_TOPIC, 100, gnss_ephem_callback);
        sub_glo_ephem = n.subscribe(GNSS_GLO_EPHEM_TOPIC, 100, gnss_glo_ephem_callback);
        sub_gnss_meas = n.subscribe(GNSS_MEAS_TOPIC, 100, gnss_meas_callback);
        sub_gnss_iono_params = n.subscribe(GNSS_IONO_PARAMS_TOPIC, 100, gnss_iono_params_callback);

        if (GNSS_LOCAL_ONLINE_SYNC)
        {
            sub_gnss_time_pluse_info = n.subscribe(GNSS_TP_INFO_TOPIC, 100, 
                gnss_tp_info_callback);
            sub_local_trigger_info = n.subscribe(LOCAL_TRIGGER_INFO_TOPIC, 100, 
                local_trigger_info_callback);
        }
        else
        {
            time_diff_gnss_local = GNSS_LOCAL_TIME_DIFF;
            estimator.inputGNSSTimeDiff(time_diff_gnss_local);
            time_diff_valid = true;
        }
    }

    ros::Subscriber sub_encoder;
    if (ENCODER_ENABLE) 
        sub_encoder = n.subscribe(ENCODER_TOPIC, 2000, encoder_callback, ros::TransportHints().tcpNoDelay());

    thread sync_thread{stereo_sync};
    ros::spin();

    return 0;
}
