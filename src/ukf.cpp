#include "ukf.h"
#include "ukf_impl.h"

#include <iostream>

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (is_initialized_) {
    auto delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;
    Prediction(delta_t);
    switch (meas_package.sensor_type_) {
      case MeasurementPackage::LASER:
        UpdateLidar(meas_package);
        break;
      case MeasurementPackage::RADAR:
        UpdateRadar(meas_package);
        break;
    }
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    x_(2) = 0;
    x_(3) = 0;
    x_(4) = 0;

    P_ = Eigen::Matrix<double, 5, 5>::Identity();
    P_(0, 0) = std_laspx_ * std_laspx_;
    P_(1, 1) = std_laspy_ * std_laspy_;

    // express that we are very uncertain about the velocity
    P_(2, 2) = 10;
    P_(3, 3) = 0.01;
    P_(4, 4) = 0.01;

    is_initialized_ = true;
  }
  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  Predictor<5, 1, 2> predictor =
      [](Eigen::Matrix<double, 5, 1>& state,
         const Eigen::Matrix<double, 1, 1>& control,
         const Eigen::Matrix<double, 2, 1>& process_noise) {
        auto dt = control(0);
        auto nu_a = process_noise(0);
        auto nu_phi = process_noise(1);
        auto x = state(0);
        auto y = state(1);
        auto v = state(2);
        auto phi = state(3);
        auto phi_dot = state(4);

        auto dt_2 = dt * dt * 0.5;
        auto cos_phi = std::cos(phi);
        auto sin_phi = std::sin(phi);

        state(0) += dt_2 * cos_phi * nu_a;
        state(1) += dt_2 * sin_phi * nu_a;
        state(2) += dt * nu_a;
        state(3) += dt * phi_dot + dt_2 * nu_phi;
        state(4) += dt * nu_phi;

        if (std::fabs(phi_dot) < std::numeric_limits<double>::epsilon()) {
          auto dtv = dt * v;
          state(0) += dtv * cos_phi;
          state(1) += dtv * sin_phi;
        } else {
          auto v_div_phi = v / phi_dot;
          auto dtp = phi_dot * dt;
          state(0) += v_div_phi * (std::sin(phi + dtp) - sin_phi);
          state(1) += v_div_phi * (-std::cos(phi + dtp) + cos_phi);
        }
      };

  Eigen::Matrix<double, 1, 1> control;
  control(0) = delta_t;
  Eigen::Matrix<double, 2, 2> process_noise_cov =
      Eigen::Matrix<double, 2, 2>::Identity();
  process_noise_cov(0, 0) = std_a_ * std_a_;
  process_noise_cov(1, 1) = std_yawdd_ * std_yawdd_;

  Xsig_pred_ = ukfPredict(x_, P_, predictor, control, process_noise_cov);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  if (not use_laser_) {
    return;
  }

  MeasurementFunction<2, 5> meas_func =
      [](Eigen::Matrix<double, 2, 1>& meas,
         const Eigen::Matrix<double, 5, 1>& state) {
        meas(0) = state(0);
        meas(1) = state(1);
      };

  Eigen::Matrix<double, 2, 1> meas;
  meas(0) = meas_package.raw_measurements_(0);
  meas(1) = meas_package.raw_measurements_(1);
  Eigen::Matrix<double, 2, 2> meas_noise_cov =
      Eigen::Matrix<double, 2, 2>::Identity();
  meas_noise_cov(0, 0) = std_laspx_ * std_laspx_;
  meas_noise_cov(1, 1) = std_laspy_ * std_laspy_;

  ukfUpdate(x_, P_, Xsig_pred_, meas_func, meas, meas_noise_cov);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  if (not use_radar_) {
    return;
  }

  MeasurementFunction<3, 5> meas_func =
      [](Eigen::Matrix<double, 3, 1>& meas,
         const Eigen::Matrix<double, 5, 1>& state) {
        auto x = state(0);
        auto y = state(1);
        auto v = state(2);
        auto phi = state(3);

        auto vx = v * cos(phi);
        auto vy = v * sin(phi);

        meas(0) = std::sqrt(x * x + y * y);
        meas(1) = std::atan2(y, x);

        if (meas(0) < std::numeric_limits<double>::epsilon()) {
          meas(2) = v;
        } else {
          meas(2) = (x * vx + y * vy) / meas(0);
        }
      };

  Eigen::Matrix<double, 3, 1> meas;
  meas(0) = meas_package.raw_measurements_(0);
  meas(1) = meas_package.raw_measurements_(1);
  meas(2) = meas_package.raw_measurements_(2);

  Eigen::Matrix<double, 3, 3> meas_noise_cov =
      Eigen::Matrix<double, 3, 3>::Identity();
  meas_noise_cov(0, 0) = std_radr_ * std_radr_;
  meas_noise_cov(1, 1) = std_radphi_ * std_radphi_;
  meas_noise_cov(2, 2) = std_radrd_ * std_radrd_;

  ukfUpdate(x_, P_, Xsig_pred_, meas_func, meas, meas_noise_cov);
}