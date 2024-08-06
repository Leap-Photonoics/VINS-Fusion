#include "vpi_helper.h"

void vpiCheckState(const VPIStatus &status)
{
    if (status != VPI_SUCCESS)                            
    {                                                       
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
        vpiGetLastStatusMessage(buffer, sizeof(buffer));
        ROS_ERROR_STREAM(vpiStatusGetName(status) << ": " << buffer);
        ROS_BREAK();
    }     
}

void convertVPIArrayToCV(const VPIArray &vpi_array, vector<cv::Point2f> &cv_array)
{
    cv_array.clear();
    VPIArrayData array_data;
    vpiCheckState(vpiArrayLockData(vpi_array, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &array_data));
    VPIArrayBufferAOS &aos = array_data.buffer.aos;
    cv_array.reserve(*aos.sizePointer);
    VPIKeypointF32 *data = reinterpret_cast<VPIKeypointF32*>(aos.data);
    for (int i = 0; i < *aos.sizePointer; i++)
        cv_array.push_back(cv::Point2f(data[i].x, data[i].y));
    vpiCheckState(vpiArrayUnlock(vpi_array));
}

void convertVPIArrayToCV(const VPIArray &vpi_array, vector<uint8_t> &cv_array)
{
    cv_array.clear();
    VPIArrayData array_data;
    vpiCheckState(vpiArrayLockData(vpi_array, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &array_data));
    VPIArrayBufferAOS &aos = array_data.buffer.aos;
    cv_array.reserve(*aos.sizePointer);
    uint8_t *data = reinterpret_cast<uint8_t*>(aos.data);
    for (int i = 0; i < *aos.sizePointer; i++)
        cv_array.push_back(data[i]);
    vpiCheckState(vpiArrayUnlock(vpi_array));
}

void convertCVtoVPIArray(const vector<cv::Point2f> &cv_array, VPIArray &vpi_array)
{
    VPIArrayData array_data;
    vpiCheckState(vpiArraySetSize(vpi_array, cv_array.size()));
    vpiCheckState(vpiArrayLockData(vpi_array, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &array_data));
    VPIArrayBufferAOS &aos = array_data.buffer.aos;
    VPIKeypointF32 *data = reinterpret_cast<VPIKeypointF32*>(aos.data);
    for (int i = 0; i < cv_array.size(); i++)
    {
        data[i].x = cv_array[i].x;
        data[i].y = cv_array[i].y;
    }
    vpiCheckState(vpiArrayUnlock(vpi_array));
}

void copyVPIArray(const VPIArray &src, VPIArray &dst)
{
    VPIArrayData src_data, dst_data;
    int size;
    vpiCheckState(vpiArrayGetSize(src, &size));
    vpiCheckState(vpiArraySetSize(dst, size));
    VPIArrayType type;
    size_t item_size;
    vpiCheckState(vpiArrayGetType(src, &type));
    if (type == VPI_ARRAY_TYPE_KEYPOINT_F32)
        item_size = sizeof(VPIKeypointF32);
    else if (type == VPI_ARRAY_TYPE_U8)
        item_size = sizeof(uint8_t);
    else
        ROS_ERROR("Unsupported array type");
    vpiCheckState(vpiArrayLockData(src, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &src_data));
    vpiCheckState(vpiArrayLockData(dst, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &dst_data));
    VPIArrayBufferAOS &src_aos = src_data.buffer.aos;
    VPIArrayBufferAOS &dst_aos = dst_data.buffer.aos;
    memcpy(dst_aos.data, src_aos.data, item_size * size);
    vpiCheckState(vpiArrayUnlock(src));
    vpiCheckState(vpiArrayUnlock(dst));
}
