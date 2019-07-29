#include "catch.hpp"
#include "comparison_helpers.hpp"

#include "reor/gpnh_dar_regularizer.hpp"
#include "reor/numerics_helpers.hpp"

#include <iostream>

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

namespace {

struct GPNH_DAR_eigen_host :
      public GPNH_DAR_regularizer<backends::Eigen_backend<double> > {};

} // anonymous namespace

TEST_CASE("Test GPNH-DAR regularization with Eigen matrices",
          "[matrix_factorizations][gpnh_dar_regularizer][eigen_backend]")
{
   SECTION("Throws when given negative dictionary regularization parameter")
   {
      GPNH_DAR_eigen_host host_instance;
      CHECK_THROWS(host_instance.set_epsilon_states(-1));
   }

   SECTION("Throws when given negative weights regularization parameters")
   {
      GPNH_DAR_eigen_host host_instance;
      CHECK_THROWS(host_instance.set_epsilon_weights(-1));
      CHECK_THROWS(host_instance.set_eta_weights(-1));
   }

   SECTION("Throws when given given negative parameters regularization parameter")
   {
      GPNH_DAR_eigen_host host_instance;
      CHECK_THROWS(host_instance.set_epsilon_parameters(-1));
   }

   SECTION("Returns zero penalty when all regularization parameters vanish")
   {
      const int n_features = 34;
      const int n_samples = 450;
      const int n_components = 18;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({1, 2});
      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 0;
      const double test_epsilon_weights = 0;
      const double test_eta_weights = 0;
      const double test_epsilon_parameters = 0;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      const double penalty = host_instance.penalty(X, S, Gamma);
      const double expected_penalty = 0;

      CHECK(is_equal(penalty, expected_penalty));
   }

   SECTION("Returns expected dictionary penalty for vanishing weights regularization")
   {
      const int n_features = 3;
      const int n_samples = 321;
      const int n_components = 5;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Zero(n_components, n_samples));

      const std::vector<int> lag_set({1, 2, 3});

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 3.;
      const double test_epsilon_weights = 0;
      const double test_eta_weights = 0;
      const double test_epsilon_parameters = 0;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      const double penalty = host_instance.penalty(X, S, Gamma);
      double expected_penalty = 0;
      for (int i = 0; i < n_features; ++i) {
         for (int k1 = 0; k1 < n_components; ++k1) {
            for (int k2 = 0; k2 < n_components; ++k2) {
               expected_penalty += (S(i, k1) - S(i, k2)) *
                  (S(i, k1) - S(i, k2));
            }
         }
      }
      expected_penalty *= test_epsilon_states /
         (n_features * n_components * (n_components - 1));

      CHECK(is_equal(penalty, expected_penalty, tolerance));
   }

   SECTION("Returns expected weights norm penalty when all other regularizations vanish")
   {
      const int n_features = 6;
      const int n_samples = 31;
      const int n_components = 2;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({1, 2, 3, 5});

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 0;
      const double test_epsilon_weights = 0;
      const double test_eta_weights = 2.;
      const double test_epsilon_parameters = 0;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      const double penalty = host_instance.penalty(X, S, Gamma);
      double expected_penalty = 0.5 * test_eta_weights * Gamma.squaredNorm();

      std::cout << "penalty = " << penalty << '\n';
      std::cout << "expected = " << expected_penalty << '\n';
      CHECK(is_equal(penalty, expected_penalty, tolerance));
   }
}

#endif
