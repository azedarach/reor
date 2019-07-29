#include "catch.hpp"
#include "comparison_helpers.hpp"

#include "reor/numerics_helpers.hpp"
#include "reor/trivial_matrix_factorization_regularizer.hpp"

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

TEST_CASE("Test trivial regularization with Eigen matrices",
          "[matrix_factorizations][trivial_matrix_factorization_regularizer][eigen_backend]")
{
   using Regularization = Trivial_matrix_factorization_regularizer;

   SECTION("Returns zero for arbitrary matrices")
   {
      const int n_features = 10;
      const int n_samples = 100;
      const int n_components = 4;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const double penalty = Regularization::penalty(X, S, Gamma);
      const double expected_penalty = 0;

      CHECK(is_equal(penalty, expected_penalty));
   }

   SECTION("Returns zero dictionary jacobian for arbitrary matrices")
   {
      const int n_features = 41;
      const int n_samples = 200;
      const int n_components = 10;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_S(Eigen::MatrixXd::Random(n_features, n_components));

      REQUIRE(!is_equal(jac_S,
                        Eigen::MatrixXd::Zero(n_features, n_components)));

      Regularization::dictionary_gradient(X, S, Gamma, jac_S);

      CHECK(is_equal(jac_S, Eigen::MatrixXd::Zero(n_features, n_components)));
   }

   SECTION("Returns zero weights jacobian for arbitrary matrices")
   {
      const int n_features = 32;
      const int n_samples = 700;
      const int n_components = 17;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_Gamma(
         Eigen::MatrixXd::Random(n_components, n_samples));

      REQUIRE(!is_equal(jac_Gamma,
                        Eigen::MatrixXd::Zero(n_components, n_samples)));

      Regularization::weights_gradient(X, S, Gamma, jac_Gamma);

      CHECK(is_equal(jac_Gamma,
                     Eigen::MatrixXd::Zero(n_components, n_samples)));
   }
}

#endif
