#include "catch.hpp"

#include "reor/l2_spa.hpp"
#include "reor/numerics_helpers.hpp"

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

TEST_CASE("Test cost function with no regularization and Eigen matrices",
          "[l2_spa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;

   SECTION("Returns zero for perfect reconstruction")
   {
      const int n_features = 5;
      const int n_components = 3;
      const int n_samples = 30;
      const double tolerance = 1e-14;

      const Eigen::MatrixXd S(
         Eigen::MatrixXd::Random(n_features, n_components));

      Eigen::MatrixXd Gamma(
         Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         for (int k = 0; k < n_components; ++k) {
            Gamma(k, i) /= col_sums(i);
         }
      }

      const Eigen::MatrixXd X(S * Gamma);

      L2_SPA<Backend> spa(X, S, Gamma);

      const double cost = spa.cost();
      const double expected_cost = 0.;

      CHECK(is_equal(cost, expected_cost, tolerance));
   }
}

#endif
