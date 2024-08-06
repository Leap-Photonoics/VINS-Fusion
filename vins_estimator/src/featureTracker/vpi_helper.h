#pragma once

#include <vpi/Array.h>
#include <vector>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

using namespace std;

void vpiCheckState(const VPIStatus &status);
void convertVPIArrayToCV(const VPIArray &vpi_array, vector<cv::Point2f> &cv_array);
void convertVPIArrayToCV(const VPIArray &vpi_array, vector<uint8_t> &cv_array);
void convertCVtoVPIArray(const vector<cv::Point2f> &cv_array, VPIArray &vpi_array);
void copyVPIArray(const VPIArray &src, VPIArray &dst);
