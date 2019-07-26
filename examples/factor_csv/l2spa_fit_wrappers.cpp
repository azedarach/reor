#include "l2spa_fit_wrappers.hpp"

namespace reor {

double calculate_rss(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
   const Eigen::MatrixXd residuals(A - B);
   return residuals.cwiseProduct(residuals).sum();
}

double calculate_rmse(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
   const Eigen::MatrixXd residuals(A - B);
   const double mse = residuals.cwiseProduct(residuals).mean();
   return std::sqrt(mse);
}

} // namespace reor
