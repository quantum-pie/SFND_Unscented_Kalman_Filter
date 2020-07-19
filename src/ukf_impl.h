#pragma once

#include "Eigen/Cholesky"
#include "Eigen/Dense"

#include <vector>

using namespace Eigen;

/// @brief Alias for prediction functor
template <int N, int C, int W>
using Predictor =
    std::function<void(Matrix<double, N, 1>&, const Matrix<double, C, 1>&,
                       const Matrix<double, W, 1>&)>;

/// @brief Alias for measurement functor
template <int N, int C>
using MeasurementFunction =
    std::function<void(Matrix<double, N, 1>&, const Matrix<double, C, 1>&)>;

template <int DIM>
Matrix<double, DIM, 2 * DIM> drawSigmaPoints() {
  // sqrt(lambda + D) where lambda = D - 3
  auto W = std::sqrt(2 * DIM - 3);
  Matrix<double, DIM, 2 * DIM> sigma_pts;
  sigma_pts << Matrix<double, DIM, DIM>::Identity() * W,
      Matrix<double, DIM, DIM>::Identity() * -W;
  return sigma_pts;
}

/// @brief Transform initial set of sigma points to the set for distribution with given mean and covariance 
/// @tparam D dimensionality of augmented distribution 
/// @param sigma matrix with initial sigma points in columns 
/// @param mean mean of augmented state 
/// @param aug_cov_sqrt square root of noise-augmented covariance matrix 
/// @return array of sigma points
template <int D>
std::array<Matrix<double, D, 1>, 2 * D> transformSigmaPoints(
    const Matrix<double, D, 2 * D>& sigma, const Matrix<double, D, 1>& mean,
    const Matrix<double, D, D>& aug_cov_sqrt) {
  static_assert(D > 0, "Distribution dimensionality must be > 0");

  std::array<Matrix<double, D, 1>, 2 * D> result;

  // centered sigma points, scaled by covariance square root
  auto tf_sigma_pts = (aug_cov_sqrt * sigma).eval();
  for (int i = 0; i < 2 * D; ++i) {
    // add mean
    result[i] = tf_sigma_pts.col(i) + mean;
  }
  return result;
}

/// @brief Predict function
/// @tparam N size of state
/// @tparam C size of control input vector
/// @tparam W size of process noise vector
/// @tparam D size of augmented state vector
/// @param state [in, out] state
/// @param state_cov [in, out] state covariance matrix
/// @param predictor function which predicts state, given control input and process noise vector
/// @param control control vector
/// @param process_noise_cov process noise covariance
/// @return propagated sigma points array
template <int N, int C, int W, int D = N + W>
std::array<Matrix<double, N, 1>, 2 * D + 1> ukfPredict(
    Matrix<double, N, 1>& state, Matrix<double, N, N>& state_cov,
    Predictor<N, C, W> predictor, const Matrix<double, C, 1>& control,
    const Matrix<double, W, W>& process_noise_cov) {
  static_assert(N > 0, "Size of state must be > 0");
  static_assert(C > 0, "Size of control vector must be > 0");
  static_assert(W > 0, "Size of noise vector must be > 0");

  // draw points once (depend only on the dimensionality)
  static const auto sigma_pts_init = drawSigmaPoints<D>();

  // lambda / (lambda + D) where lambda = D - 3
  static constexpr auto w0 = static_cast<double>(D - 3) / (2 * D - 3);

  // 1 / (2*(lambda + D)) where lambda = D - 3
  static constexpr auto w = 1.0 / (4 * D - 6);

  // propagate mean (call predictor without noise)
  auto mean_state = state;
  predictor(mean_state, control, Matrix<double, W, 1>::Zero());

  // block-diagonal augmented covariance
  Matrix<double, D, D> aug_cov = Matrix<double, D, D>::Zero();
  aug_cov.template block<N, N>(0, 0) = state_cov;
  aug_cov.template block<W, W>(N, N) = process_noise_cov;

  // calculate covariance square root using Cholesky decomposition
  LLT<Matrix<double, D, D>> aug_cov_llt(aug_cov);
  Matrix<double, D, D> aug_cov_sqrt = aug_cov_llt.matrixL();

  // fit sigma points to the current augmented state distribution
  Matrix<double, D, 1> aug_state = Matrix<double, D, 1>::Zero();
  aug_state.template head<N>() = state;
  auto sigma_points =
      transformSigmaPoints(sigma_pts_init, aug_state, aug_cov_sqrt);

  state.setZero();

  // propagated sigma points
  std::array<Matrix<double, N, 1>, 2 * D + 1> prop_points;
  for (size_t i = 0; i < 2 * D; ++i) {
    const auto& pt = sigma_points[i];

    // extract state and noise parts from sigma point
    auto state_sigma = pt.template head<N>().eval();
    auto noise_sigma = pt.template tail<W>().eval();

    // predict given state and noise
    predictor(state_sigma, control, noise_sigma);

    // accumulate state and save propagated point
    state += state_sigma;
    prop_points[i] = state_sigma;
  }

  // save propagated mean
  prop_points[2 * D] = mean_state;

  // propagated state
  state = w0 * mean_state + w * state;

  // reset covariance
  state_cov.setZero();
  for (const auto& pt : prop_points) {
    Matrix<double, N, 1> diff = pt - state;
    state_cov += diff * diff.transpose();
  }

  // propagated covariance
  state_cov = w0 * (mean_state - state) * (mean_state - state).transpose() +
              w * state_cov;

  // return propagated points to reuse in update function
  return prop_points;
}

/// @brief UKF update function
/// @tparam N size of state
/// @tparam C size of measurements vector
/// @tparam W size of measurement noise
/// @param P number of sigma points
/// @param state [in, out] state
/// @param state_cov [in, out] state covariance matrix
/// @param propagated_sigma array of propagfated sigma points
/// @param measurement_function function which predicts measurement, given state and measurement noise vector
/// @param measurements measurement vector
/// @param measurement_noise_cov measurement noise covariance matrix
template <int N, int C, int W, size_t P>
void ukfUpdate(Matrix<double, N, 1>& state, Matrix<double, N, N>& state_cov,
               const std::array<Matrix<double, N, 1>, P>& propagated_sigma,
               MeasurementFunction<C, N> measurement_function,
               const Matrix<double, C, 1>& measurements,
               const Matrix<double, W, W>& measurement_noise_cov) {
  static_assert(N > 0, "Size of state must be > 0");
  static_assert(C > 0, "Size of measurement vector must be > 0");
  static_assert(W > 0, "Size of noise vector must be > 0");

  // Infer dimensionality from number of sigma points
  static constexpr int D = (P - 1) / 2;
  using Measurement = Matrix<double, C, 1>;

  // lambda / (lambda + D) where lambda = D - 3
  static constexpr auto w0 = static_cast<double>(D - 3) / (2 * D - 3);

  // 1 / (2*(lambda + D)) where lambda = D - 3
  static constexpr auto w = 1.0 / (4 * D - 6);

  // calculate mean measurement
  Measurement mean_meas;
  measurement_function(mean_meas, state);

  // predict measurement
  Measurement avg_meas = Measurement::Zero();
  std::array<Measurement, 2 * D> sample_meas;
  for (size_t i = 0; i < 2 * D; ++i) {
    // calculate measurement for sigma-point
    measurement_function(sample_meas[i], propagated_sigma[i]);
    avg_meas += sample_meas[i];
  }

  // predicted measurement
  avg_meas = w0 * mean_meas + w * avg_meas;

  Matrix<double, N, C> state_meas_cov = Matrix<double, N, C>::Zero();
  Matrix<double, C, C> meas_cov = Matrix<double, C, C>::Zero();
  for (size_t i = 0; i < 2 * D; ++i) {
    Matrix<double, N, 1> state_diff = propagated_sigma[i] - state;
    Measurement meas_diff = sample_meas[i] - avg_meas;

    state_meas_cov += state_diff * meas_diff.transpose();
    meas_cov += meas_diff * meas_diff.transpose();
  }

  // state-measurement covariance
  state_meas_cov = w0 * (propagated_sigma[2 * D] - state) *
                       (mean_meas - avg_meas).transpose() +
                   w * state_meas_cov;

  // measurement covariance
  meas_cov = w0 * (mean_meas - avg_meas) * (mean_meas - avg_meas).transpose() +
             w * meas_cov + measurement_noise_cov;

  auto gain = (state_meas_cov * meas_cov.inverse()).eval();
  auto innovation = (gain * (measurements - avg_meas)).eval();

  // update state
  state += innovation;

  // update covariance
  state_cov -= gain * meas_cov * gain.transpose();
}
