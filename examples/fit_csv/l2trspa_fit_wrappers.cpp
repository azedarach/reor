#include "l2trspa_fit_wrappers.hpp"

namespace reor {

double calculate_rss(const Eigen::MatrixXd& residuals)
{
   return residuals.cwiseProduct(residuals).sum();
}

double calculate_rss(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
   const Eigen::MatrixXd residuals(A - B);
   return calculate_rss(residuals);
}

double calculate_rmse(const Eigen::MatrixXd& residuals)
{
   return std::sqrt(residuals.cwiseProduct(residuals).mean());
}

double calculate_rmse(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
   const Eigen::MatrixXd residuals(A - B);
   return calculate_rmse(residuals);
}

Eigen::MatrixXd get_predicted_weights(
   const Eigen::MatrixXd& observed_weights,
   const std::vector<int>& lag_set, const Eigen::MatrixXd& parameters,
   int max_horizon)
{
   const int n_components = observed_weights.rows();
   const int n_samples = observed_weights.cols();
   const int n_lags = lag_set.size();

   int max_lag = -1;
   for (auto l : lag_set) {
      if (l > max_lag) {
         max_lag = l;
      }
   }

   if (n_samples < max_lag) {
      throw std::runtime_error(
         "insufficient data for forecast");
   }

   Eigen::MatrixXd forecast(n_components, max_horizon);

   for (int h = 0; h < max_horizon; ++h) {
      const int t_target = n_samples + h;
      for (int i = 0; i < n_components; ++i) {
         Eigen::VectorXd lagged_values(n_lags);
         for (int l = 0; l < n_lags; ++l) {
            const int t_lag = t_target - lag_set[l];
            if (t_lag < n_samples) {
               lagged_values(l) = observed_weights(i, t_lag);
            } else {
               lagged_values(l) = forecast(i, t_lag - n_samples);
            }
         }
         forecast(i, h) = parameters.row(i).dot(lagged_values);
      }
   }

   return forecast;
}

Eigen::MatrixXd calculate_prediction_errors(
   const Eigen::MatrixXd& observed_weights,
   const Eigen::MatrixXd& target_data,
   const std::vector<int>& lag_set, const Eigen::MatrixXd& parameters,
   const Eigen::MatrixXd& dictionary)
{
   const int max_horizon = target_data.cols();

   const Eigen::MatrixXd forecasted_weights =
      get_predicted_weights(observed_weights, lag_set, parameters,
                            max_horizon);
   const Eigen::MatrixXd forecasted_values =
      dictionary * forecasted_weights;
   std::cout << "weights = " << forecasted_weights << '\n';
   std::cout << "col_sums = " << forecasted_weights.colwise().sum() << '\n';
   std::cout << "forecasted = " << forecasted_values << '\n';
   std::cout << "observed = " << target_data << '\n';
   return target_data - forecasted_values;
}

} // namespace reor
