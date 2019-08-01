#include "catch.hpp"

#include "reor/furthest_sum.hpp"

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

TEST_CASE("Test FurthestSum with Eigen matrices",
          "[furthest_sum][eigen_backend]")
{
   SECTION("Throws error when given non-square dissimilarity matrix")
   {
      const int n_features = 10;
      const int n_samples = 20;
      const int n_components = 2;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));

      CHECK_THROWS(furthest_sum(X, n_components, 0));
   }
}

#endif
