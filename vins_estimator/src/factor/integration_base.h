#pragma once

#include "../utility/utility.h"
#include "../estimator/parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;
using namespace std;

class IntegrationBase
{
  public:
    IntegrationBase() = delete;
    IntegrationBase(const Vector3d &_acc_0, const Vector3d &_gyr_0,
                    const Vector3d &_linearized_ba, const Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Matrix<double, 15, 15>::Identity()}, covariance{Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Vector3d::Zero()}, delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()}

    {
        noise = Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Matrix3d::Identity();
    }

    // 另写一个带Encoder的构造函数
    IntegrationBase (const Vector3d &_acc_0, const Vector3d &_gyr_0,
                     const Vector3d &_linearized_ba, const Vector3d &_linearized_bg,
                     const Matrix<double, 6, 1> &_enc_v_0)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, enc_v_0{_enc_v_0},
          linearized_acc{_acc_0}, linearized_gyr{_gyr_0}, linearized_enc_v{_enc_v_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
          jacobian_enc{Matrix<double, 21, 21>::Identity()}, covariance_enc{Matrix<double, 21, 21>::Zero()},
          sum_dt{0.0}, delta_p{Vector3d::Zero()}, delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()},
          delta_eta{Matrix<double, 6, 1>::Zero()}
    {
        noise_enc = Matrix<double, 30, 30>::Zero();
        noise_enc.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(6, 6) = (ENC_N * ENC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(9, 9) = (ENC_N * ENC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(12, 12) = (ACC_N * ACC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(15, 15) = (GYR_N * GYR_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(18, 18) = (ENC_N * ENC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(21, 21) = (ENC_N * ENC_N) * Matrix3d::Identity();
        noise_enc.block<3, 3>(24, 24) = (ACC_W * ACC_W) * Matrix3d::Identity();
        noise_enc.block<3, 3>(27, 27) = (GYR_W * GYR_W) * Matrix3d::Identity();
    }

    void push_back(double dt, const Vector3d &acc, const Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void push_back(double dt, const Vector3d &acc, const Vector3d &gyr, const Matrix<double, 6, 1> &enc_v)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        enc_v_buf.push_back(enc_v);
        propagate(dt, acc, gyr, enc_v);
    }

    void repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        if (ENCODER_ENABLE)
        {
            enc_v_0 = linearized_enc_v;
            delta_eta.setZero();
            jacobian_enc.setIdentity();
            covariance_enc.setZero();
            for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
                propagate(dt_buf[i], acc_buf[i], gyr_buf[i], enc_v_buf[i]);
        }
        else
            for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
                propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        
    }

    void midPointIntegration(double _dt, 
                            const Vector3d &_acc_0, const Vector3d &_gyr_0,
                            const Vector3d &_acc_1, const Vector3d &_gyr_1,
                            const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                            const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                            Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                            Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian)
    {
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            step_jacobian = F;
            step_V = V;
            
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }

    void midPointIntegration(double _dt,
                            const Vector3d &_acc_0, const Vector3d &_gyr_0, const Matrix<double, 6, 1> &_enc_v_0,
                            const Vector3d &_acc_1, const Vector3d &_gyr_1, const Matrix<double, 6, 1> &_enc_v_1,
                            const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                            const Matrix<double, 6, 1> &delta_eta, const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                            Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                            Matrix<double, 6, 1> &result_delta_eta, Vector3d &result_linearized_ba, Vector3d &result_linearized_bg,
                            bool update_jacobian)
    {
        // 中值法
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        // Matrix<double, 6, 1> un_enc_0 = delta_q * RIO * _enc_v_0; // encoder
        // Matrix<double, 6, 1> un_enc_1 = result_delta_q * RIO * _enc_v_1; // encoder
        // result_delta_eta = delta_eta + 0.5 * (un_enc_0 + un_enc_1) * _dt; // encoder
        result_delta_eta.block<3, 1>(0, 0) = 0.5 * _dt * (delta_q * RIO * _enc_v_0.block<3, 1>(0, 0) + result_delta_q * RIO * _enc_v_1.block<3, 1>(0, 0));
        result_delta_eta.block<3, 1>(3, 0) = 0.5 * _dt * (delta_q * RIO * _enc_v_0.block<3, 1>(3, 0) + result_delta_q * RIO * _enc_v_1.block<3, 1>(3, 0));
        result_delta_eta += delta_eta;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;
        

        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Vector3d e_0_x_l = RIO * _enc_v_0.block<3, 1>(0, 0), e_0_x_r = RIO * _enc_v_0.block<3, 1>(3, 0);
            Vector3d e_1_x_l = RIO * _enc_v_1.block<3, 1>(0, 0), e_1_x_r = RIO * _enc_v_1.block<3, 1>(3, 0);
            Matrix3d R_w_x = Utility::skewSymmetric(w_x), 
                    R_a_0_x = Utility::skewSymmetric(a_0_x), 
                    R_a_1_x = Utility::skewSymmetric(a_1_x),
                    R_e_0_x_l = Utility::skewSymmetric(e_0_x_l), R_e_0_x_r = Utility::skewSymmetric(e_0_x_r),
                    R_e_1_x_l = Utility::skewSymmetric(e_1_x_l), R_e_1_x_r = Utility::skewSymmetric(e_1_x_r);

            MatrixXd F = MatrixXd::Zero(21, 21);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 15) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 18) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 18) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 15) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 18) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;

            F.block<3, 3>(9, 3) = -0.5 * delta_q.toRotationMatrix() * R_e_0_x_l * _dt + 
                                -0.5 * result_delta_q.toRotationMatrix() * R_e_1_x_l * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(9, 15) = 0.5 * result_delta_q.toRotationMatrix() * R_e_1_x_l * _dt * _dt;
            
            F.block<3, 3>(12, 3) = -0.5 * delta_q.toRotationMatrix() * R_e_0_x_r * _dt + 
                                -0.5 * result_delta_q.toRotationMatrix() * R_e_1_x_r * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            F.block<3, 3>(12, 15) = 0.5 * result_delta_q.toRotationMatrix() * R_e_1_x_r * _dt * _dt;

            F.block<3, 3>(15, 15) = Matrix3d::Identity();
            F.block<3, 3>(18, 18) = Matrix3d::Identity();

            MatrixXd V = MatrixXd::Zero(21, 30);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 12) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 15) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 15) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 12) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 15) =  V.block<3, 3>(6, 3);

            V.block<3, 3>(9, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_e_1_x_l * _dt * _dt; // 相差-
            V.block<3, 3>(9, 6) = 0.5 * delta_q.toRotationMatrix() * RIO * _dt;
            V.block<3, 3>(9, 15) = V.block<3, 3>(9, 3);
            V.block<3, 3>(9, 18) = 0.5 * result_delta_q.toRotationMatrix() * RIO * _dt;

            V.block<3, 3>(12, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_e_1_x_r * _dt * _dt; // 相差-
            V.block<3, 3>(12, 9) = 0.5 * delta_q.toRotationMatrix() * RIO * _dt;
            V.block<3, 3>(12, 15) = V.block<3, 3>(12, 3);
            V.block<3, 3>(12, 21) = 0.5 * result_delta_q.toRotationMatrix() * RIO * _dt;

            V.block<3, 3>(15, 24) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(18, 27) = MatrixXd::Identity(3,3) * _dt;

            // step_jacobian_enc = F;
            // step_V_enc = V;

            jacobian_enc = F * jacobian_enc;
            covariance_enc = F * covariance_enc * F.transpose() + V * noise_enc * V.transpose();
        }
    }

    void propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        // checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    void propagate(double _dt, const Vector3d &_acc_1, const Vector3d &_gyr_1, const Matrix<double, 6, 1> &_enc_v_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        enc_v_1 = _enc_v_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Matrix<double, 6, 1> result_delta_eta;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, enc_v_0, _acc_1, _gyr_1, _enc_v_1,
                            delta_p, delta_q, delta_v, delta_eta, linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v, result_delta_eta,
                            result_linearized_ba, result_linearized_bg, 1);

        // checkJacobian(_dt, acc_0, gyr_0, enc_v_0, acc_1, gyr_1, enc_v_1, delta_p, delta_q, delta_v, delta_eta,
        //                    linearized_ba, linearized_bg);

        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        delta_eta = result_delta_eta;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
        enc_v_0 = enc_v_1;
    }

    Matrix<double, 15, 1> evaluate(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi, const Vector3d &Bai, const Vector3d &Bgi,
                                          const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj, const Vector3d &Baj, const Vector3d &Bgj)
    {
        Matrix<double, 15, 1> residuals;

        Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Vector3d dba = Bai - linearized_ba;
        Vector3d dbg = Bgi - linearized_bg;

        Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    Matrix<double, 21, 1> evaluate_enc(const Vector3d &Pi, const Quaterniond &Qi, const Vector3d &Vi,
                                const Vector3d &Bai, const Vector3d &Bgi, 
                                const Vector3d &Pj, const Quaterniond &Qj, const Vector3d &Vj,
                                const Vector3d &Baj, const Vector3d &Bgj)
    {
        Matrix<double, 21, 1> residuals;

        Matrix3d dp_dba = jacobian_enc.block<3, 3>(0, 15);
        Matrix3d dp_dbg = jacobian_enc.block<3, 3>(0, 18);

        Matrix3d dq_dbg = jacobian_enc.block<3, 3>(3, 18);

        Matrix3d dv_dba = jacobian_enc.block<3, 3>(6, 15);
        Matrix3d dv_dbg = jacobian_enc.block<3, 3>(6, 18);

        Matrix3d do_l_dbg = jacobian_enc.block<3, 3>(9, 18);
        Matrix3d do_r_dbg = jacobian_enc.block<3, 3>(12, 18);

        Vector3d dba = Bai - linearized_ba;
        Vector3d dbg = Bgi - linearized_bg;

        Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
        Vector3d corrected_delta_eta_l = delta_eta.block<3, 1>(0, 0) + do_l_dbg * dbg;
        Vector3d corrected_delta_eta_r = delta_eta.block<3, 1>(3, 0) + do_r_dbg * dbg;
        // ROS_INFO_STREAM("do_l_dbg " << do_l_dbg);
        // ROS_INFO_STREAM("delta_eta " << delta_eta.transpose());
        // ROS_INFO_STREAM("Corrected_delta_eta " << corrected_delta_eta_l.transpose());


        residuals.block<3, 1>(0, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(3, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(6, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;

        // residuals.block<3, 1>(9, 0) = Qi.inverse() * (Pj - Pi) - TIO + Qi.inverse() * Qj * TIO - corrected_delta_eta;
        residuals.block<3, 1>(9, 0) = (Qi.inverse() * ((Pj + Qj * TIO_L) - (Pi + Qi * TIO_L)) - corrected_delta_eta_l);
        residuals.block<3, 1>(12, 0) = (Qi.inverse() * ((Pj + Qj * TIO_R) - (Pi + Qi * TIO_R)) - corrected_delta_eta_r);

        residuals.block<3, 1>(15, 0) = Baj - Bai;
        residuals.block<3, 1>(18, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Vector3d acc_0, gyr_0;
    Vector3d acc_1, gyr_1;
    Matrix<double, 6, 1> enc_v_0, enc_v_1;

    const Vector3d linearized_acc, linearized_gyr;
    const Matrix<double, 6, 1> linearized_enc_v;
    Vector3d linearized_ba, linearized_bg;

    Matrix<double, 15, 15> jacobian, covariance;
    Matrix<double, 15, 15> step_jacobian;
    Matrix<double, 15, 18> step_V;

    Matrix<double, 18, 18> noise;
    Matrix<double, 30, 30> noise_enc;
    Matrix<double, 21, 21> jacobian_enc, covariance_enc;
    Matrix<double, 21, 21> step_jacobian_enc;
    Matrix<double, 21, 30> step_V_enc;

    double sum_dt;
    Vector3d delta_p;
    Quaterniond delta_q;
    Vector3d delta_v;
    Matrix<double, 6, 1> delta_eta;

    std::vector<double> dt_buf;
    std::vector<Vector3d> acc_buf;
    std::vector<Vector3d> gyr_buf;
    std::vector<Matrix<double, 6, 1>> enc_v_buf; // encoder 


/*

    void eulerIntegration(double _dt, const Vector3d &_acc_0, const Vector3d &_gyr_0,
                            const Vector3d &_acc_1, const Vector3d &_gyr_1,
                            const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                            const Vector3d &linearized_ba, const Vector3d &linearized_bg,
                            Vector3d &result_delta_p, Quaterniond &result_delta_q, Vector3d &result_delta_v,
                            Vector3d &result_linearized_ba, Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Matrix<double, 12, 12> noise = Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }     
*/

    void checkJacobian(double _dt, const Vector3d &_acc_0, const Vector3d &_gyr_0, 
                                   const Vector3d &_acc_1, const Vector3d &_gyr_1,
                            const Vector3d &delta_p, const Quaterniond &delta_q, const Vector3d &delta_v,
                            const Vector3d &linearized_ba, const Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb p       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb q       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb v       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb ba       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb bg       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << endl;
    }
};