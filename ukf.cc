#include <Eigen/Dense>
#include <bits/stdc++.h>

void SaveData(const Eigen::MatrixXd &data, const std::string &filename) {
  std::ofstream file(filename);
  for (int i = 0; i < data.rows(); ++i)
    for (int j = 0; j < data.cols(); ++j)
      file << data(i, j) << "\n,"[j < data.cols() - 1];
  file.close();
}

int main() {
  // Step 1: Generate Simulated Measurement Data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dis(0, 1);

  constexpr int filters = 10;
  constexpr int n = 1000;
  constexpr double amplitude = 3;

  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(filters, n);
  Eigen::Matrix2d F;
  F << 1, 1, 0, 1;
  Eigen::Matrix2d Q = 0.001 * Eigen::Matrix2d::Identity();

  Eigen::Vector2d X0(0, 0.5);

  for (int i = 0; i < filters; ++i) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(2, n);

    for (int j = 0; j < n; ++j) {
      if (j == 0) {
        X.col(j) = X0;
      } else {
        X.col(j) = F * X.col(j - 1);
      }
    }

    for (int j = 0; j < n; ++j) {
      X.col(j) += Q.llt().matrixL() * Eigen::Vector2d(dis(gen), dis(gen));
    }

    for (int j = 0; j < n; ++j) {
      Z(i, j) = amplitude * sin(X(0, j) * M_PI / 180.) + sqrt(0.5) * dis(gen);
    }
  }

  SaveData(Z, "data.csv");

  // Step 2: Unscented Kalman filter
  Eigen::MatrixXd Z_ests = Eigen::MatrixXd::Zero(filters, n);
  Eigen::VectorXd Z_fusions = Eigen::VectorXd::Zero(n);
  constexpr int l = 2; // dimension, sigma points: 2l + 1
  constexpr double alpha = 0.0025;
  constexpr double beta = 2.0;
  constexpr double kappa = 0.0;
  constexpr double lambda = alpha * alpha * (l + kappa) - l;
  constexpr double ws0 = lambda / (l + lambda);
  constexpr double wsi = 0.5 / (l + lambda);
  constexpr double wc0 = lambda / (l + lambda) + (1 - alpha * alpha + beta);
  constexpr double wci = 0.5 / (l + lambda);

  std::vector<Eigen::Vector2d> X_ests(filters, X0);
  std::vector<Eigen::Matrix2d> P_ests(filters, Eigen::Matrix2d::Identity());

  for (int i = 0; i < n; ++i) {
    for (int filter = 0; filter < filters; ++filter) {
      auto &X_est = X_ests[filter];
      auto &P_est = P_ests[filter];

      // Predict Process
      // 1.1 Generate sigma points
      Eigen::MatrixXd x_sigmas = Eigen::MatrixXd::Zero(2, 2 * l + 1);
      x_sigmas.col(0) = X_est;
      Eigen::Matrix2d sqrt_P = P_est.llt().matrixL();
      for (int j = 0; j < l; ++j) {
        x_sigmas.col(j + 1) = X_est + std::sqrt(l + lambda) * sqrt_P.col(j);
        x_sigmas.col(j + 1 + l) = X_est - std::sqrt(l + lambda) * sqrt_P.col(j);
      }

      // 1.2 Predict mean
      Eigen::Vector2d X_pred = Eigen::Vector2d::Zero();
      for (int j = 0; j < 2 * l + 1; ++j) {
        if (j == 0) {
          X_pred += ws0 * F * x_sigmas.col(j);
        } else {
          X_pred += wsi * F * x_sigmas.col(j);
        }
      }

      // 1.3 Predict covariance
      Eigen::Matrix2d P_pred = Q;
      for (int j = 0; j < 2 * l + 1; ++j) {
        Eigen::Vector2d diff = F * x_sigmas.col(j) - X_pred;
        if (j == 0) {
          P_pred += wc0 * diff * diff.transpose();
        } else {
          P_pred += wci * diff * diff.transpose();
        }
      }

      // Update Process
      // 2.1 Generate sigma points for predict result
      Eigen::MatrixXd z_sigmas = Eigen::MatrixXd::Zero(2, 2 * l + 1);
      z_sigmas.col(0) = X_pred;
      Eigen::Matrix2d sqrt_P_pred = P_pred.llt().matrixL();
      for (int j = 0; j < l; ++j) {
        z_sigmas.col(j + 1) =
            X_pred + std::sqrt(l + lambda) * sqrt_P_pred.col(j);
        z_sigmas.col(j + 1 + l) =
            X_pred - std::sqrt(l + lambda) * sqrt_P_pred.col(j);
      }

      // 2.2 Predict measurement
      double z_pred = 0;
      for (int j = 0; j < 2 * l + 1; ++j) {
        if (j == 0) {
          z_pred += ws0 * amplitude * sin(z_sigmas(0, j) * M_PI / 180.);
        } else {
          z_pred += wsi * amplitude * sin(z_sigmas(0, j) * M_PI / 180.);
        }
      }

      // 2.3 Predict measurement covariance
      double S = 0.5;
      for (int j = 0; j < 2 * l + 1; ++j) {
        double diff = amplitude * sin(z_sigmas(0, j) * M_PI / 180.) - z_pred;
        if (j == 0) {
          S += wc0 * diff * diff;
        } else {
          S += wci * diff * diff;
        }
      }

      // 2.4 Predict cross covariance
      Eigen::Vector2d C = Eigen::Vector2d::Zero();
      for (int j = 0; j < 2 * l + 1; ++j) {
        if (j == 0) {
          C += wc0 * (F * x_sigmas.col(j) - X_pred) *
               (amplitude * sin(z_sigmas(0, j) * M_PI / 180.) - z_pred);
        } else {
          C += wci * (F * x_sigmas.col(j) - X_pred) *
               (amplitude * sin(z_sigmas(0, j) * M_PI / 180.) - z_pred);
        }
      }

      // 2.5 Kalman gain
      Eigen::Vector2d K = C / S;

      // 2.6 Update state
      X_est = X_pred + K * (Z(filter, i) - z_pred);

      // 2.7 Update covariance
      P_est = P_pred - K * S * K.transpose();

      // 3.0 Save for visualization
      Z_ests(filter, i) = amplitude * sin(X_est(0) * M_PI / 180.);
    }

    // Step 3: Fusion
    Eigen::Vector2d X_fusion = Eigen::Vector2d::Zero();
    Eigen::Matrix2d P_fusion = Eigen::Matrix2d::Zero();
    for (int filter = 0; filter < filters; ++filter) {
      P_fusion += P_ests[filter].inverse();
      X_fusion += P_ests[filter].inverse() * X_ests[filter];
    }
    P_fusion = P_fusion.inverse().eval();
    X_fusion = P_fusion * X_fusion;
    Z_fusions(i) = amplitude * sin(X_fusion(0) * M_PI / 180.);
  }

  SaveData(Z_ests, "estimate.csv");
  SaveData(Z_fusions, "fusion.csv");
}
