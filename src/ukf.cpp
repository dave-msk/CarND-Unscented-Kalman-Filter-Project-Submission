#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(n_aug_ * 2 + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  weights_.tail(2*n_aug_).setConstant(0.5/(lambda_ + n_aug_));

  NIS_laser_ = 0;
  NIS_radar_ = 0;

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  R_las_ = MatrixXd(2, 2);
  R_las_ << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  R_rad_ = MatrixXd(3, 3);
  R_rad_ << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::UKF(double std_a_, double std_yawdd_) : UKF() {
  this->std_a_ = std_a_;
  this->std_yawdd_ = std_yawdd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      double v = rho_dot;
      double yaw = phi;
      double yaw_dot = 0;
      x_ << px, py, v, yaw, yaw_dot;
    } else return;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  void (UKF::*UpdateMeasurement)(MeasurementPackage);
  switch (meas_package.sensor_type_) {
    case MeasurementPackage::LASER:
      if (!use_laser_) return;
      UpdateMeasurement = &UKF::UpdateLidar; break;
    case MeasurementPackage::RADAR:
      if (!use_radar_) return;
      UpdateMeasurement = &UKF::UpdateRadar; break;
    default:
      return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  (this->*UpdateMeasurement)(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  VectorXd x_aug(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_).setZero();
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_-n_x_, n_aug_-n_x_) << std_a_*std_a_, 0,
                                                       0, std_yawdd_*std_yawdd_;
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd Xsig(n_aug_, 2*n_aug_ + 1);
  Xsig.colwise() = x_aug;
  Xsig.block(0, 1, n_aug_, n_aug_) += sqrt(lambda_ + n_aug_)*A;
  Xsig.block(0, n_aug_+1, n_aug_, n_aug_) -= sqrt(lambda_ + n_aug_)*A;

  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double v = Xsig(2, i);
    double yaw = Xsig(3, i);
    double yaw_dot = Xsig(4, i);
    double nu_a = Xsig(5, i);
    double nu_yawdd = Xsig(6, i);

    VectorXd update(n_x_);
    if (abs(yaw_dot) < epsilon)
      update << v*cos(yaw)*delta_t,
                v*sin(yaw)*delta_t,
                0, 0, 0;
    else
      update << v/yaw_dot*(sin(yaw+yaw_dot*delta_t) - sin(yaw)),
                v/yaw_dot*(-cos(yaw+yaw_dot*delta_t) + cos(yaw)),
                0, yaw_dot*delta_t, 0;

    VectorXd noise(n_x_);
    double hdt2 = 0.5*delta_t*delta_t;
    noise << hdt2*cos(yaw)*nu_a,
             hdt2*sin(yaw)*nu_a,
             delta_t*nu_a,
             hdt2*nu_yawdd,
             delta_t*nu_yawdd;
    Xsig_pred_.col(i) = Xsig.col(i).head(n_x_) + update + noise;
  }

  x_.setZero();
  P_.setZero();
  for (int i = 0; i < 2*n_aug_+1; ++i)
    x_ += weights_(i)*Xsig_pred_.col(i);

  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    P_ += weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  MatrixXd Zsig_pred = Xsig_pred_.block(0, 0, 2, 2*n_aug_+1);
  VectorXd z_pred = VectorXd::Zero(2);
  for (int i = 0; i < 2*n_aug_+1; ++i)
    z_pred += weights_(i)*Zsig_pred.col(i);

  MatrixXd S = R_las_;
  MatrixXd T = MatrixXd::Zero(n_x_, 2);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd z_diff = Zsig_pred.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    S += weights_(i)*z_diff*z_diff.transpose();
    T += weights_(i)*x_diff*z_diff.transpose();
  }
  MatrixXd Si = S.inverse();
  MatrixXd K = T*Si;
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  x_ += K*y;
  P_ -= K*S*K.transpose();

  NIS_laser_ = y.transpose() * Si * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  MatrixXd Zsig_pred(3, 2*n_aug_+1);
  for (int i = 0; i < 2*n_aug_+1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double rho = sqrt(px*px + py*py);
    rho = (rho < epsilon) ? epsilon : rho;
    double phi = atan2(py, px);
    double rho_dot = (px*cos(yaw) + py*sin(yaw))*v/rho;

    Zsig_pred.col(i) << rho, phi, rho_dot;
  }

  VectorXd z_pred = VectorXd::Zero(3);
  for (int i = 0; i < 2*n_aug_+1; ++i)
    z_pred += weights_(i)*Zsig_pred.col(i);

  MatrixXd S = R_rad_;
  MatrixXd T = MatrixXd::Zero(n_x_, 3);

  for (int i = 0; i < 2*n_aug_+1; ++i) {
    VectorXd z_diff = Zsig_pred.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    S += weights_(i)*z_diff*z_diff.transpose();
    T += weights_(i)*x_diff*z_diff.transpose();
  }
  MatrixXd Si = S.inverse();
  MatrixXd K = T*Si;
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  x_ += K*y;
  P_ -= K*S*K.transpose();
  NIS_radar_ = y.transpose() * Si * y;
}
