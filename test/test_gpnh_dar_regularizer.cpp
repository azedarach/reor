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

      CHECK(is_equal(penalty, expected_penalty, tolerance));
   }

   SECTION("Returns expected weights residual penalty when all other regularizations vanish")
   {
      const int n_features = 5;
      const int n_samples = 4;
      const int n_components = 3;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(n_components, n_samples);

      Gamma << 0.5, 0.1, 1, 0.9,
         0.25, 0.9, 0, 0.01,
         0.25, 0, 0, 0.09;

      const std::vector<int> lag_set({1, 2});
      const std::size_t n_lags = lag_set.size();
      int max_lag = -1;
      for (auto l : lag_set) {
         if (l > max_lag) {
            max_lag = l;
         }
      }

      Eigen::MatrixXd parameters(n_components, n_lags);
      parameters << 0.1, -0.2,
         1.0, -0.3,
         0.0, 0.5;

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);
      host_instance.set_parameter_values(parameters);

      const double test_epsilon_states = 0;
      const double test_epsilon_weights = 3.5;
      const double test_eta_weights = 0;
      const double test_epsilon_parameters = 0;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      const double penalty = host_instance.penalty(X, S, Gamma);
      double expected_penalty = 0;
      for (int i = 0; i < n_components; ++i) {
         Eigen::VectorXd wi = parameters.row(i);
         for (int t = max_lag; t < n_samples; ++t) {
            double r = Gamma(i, t);
            for (std::size_t l = 0; l < n_lags; ++l) {
               r-= wi(l) * Gamma(i, t - lag_set[l]);
            }
            expected_penalty += 0.5 * test_epsilon_weights * r * r;
         }
      }

      CHECK(is_equal(penalty, expected_penalty, tolerance));
   }

   SECTION("Returns expected parameters norm penalty when all other regularizations vanish")
   {
      const int n_features = 5;
      const int n_samples = 4;
      const int n_components = 3;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({1, 2, 4});
      const int n_lags = lag_set.size();

      const Eigen::MatrixXd parameters(Eigen::MatrixXd::Random(n_components, n_lags));

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);
      host_instance.set_parameter_values(parameters);

      const double test_epsilon_states = 0;
      const double test_epsilon_weights = 0;
      const double test_eta_weights = 0;
      const double test_epsilon_parameters = 2.3;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      const double penalty = host_instance.penalty(X, S, Gamma);
      const double expected_penalty = 0.5 * test_epsilon_parameters *
         parameters.squaredNorm();

      CHECK(is_equal(penalty, expected_penalty, tolerance));
   }

   SECTION("Returns zero dictionary gradient when only one component")
   {
      const int n_features = 5;
      const int n_samples = 243;
      const int n_components = 1;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_S(Eigen::MatrixXd::Random(n_features, n_components));

      REQUIRE(jac_S.cwiseAbs().maxCoeff() > 0);

      const std::vector<int> lag_set({1});

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 1.;
      const double test_epsilon_weights = 0.5;
      const double test_eta_weights = 0.2;
      const double test_epsilon_parameters = 2;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.dictionary_gradient(X, S, Gamma, jac_S);

      const Eigen::MatrixXd expected_gradient(
         Eigen::MatrixXd::Zero(n_features, n_components));

      CHECK(is_equal(jac_S, expected_gradient, tolerance));
   }

   SECTION("Returns expected dictionary gradient when other regularizations vanish")
   {
      const int n_features = 4;
      const int n_samples = 354;
      const int n_components = 4;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_S(Eigen::MatrixXd::Random(n_features, n_components));

      const std::vector<int> lag_set({1, 2, 7});

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 1.;
      const double test_epsilon_weights = 0.5;
      const double test_eta_weights = 0.2;
      const double test_epsilon_parameters = 2;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.dictionary_gradient(X, S, Gamma, jac_S);

      Eigen::MatrixXd expected_jac_S(
         Eigen::MatrixXd::Zero(n_features, n_components));

      expected_jac_S = n_components * S
         - S * Eigen::MatrixXd::Ones(n_components, n_components);
      expected_jac_S *= 4 * test_epsilon_states /
         (n_features * n_components * (n_components - 1));

      CHECK(is_equal(jac_S, expected_jac_S, tolerance));
   }

   SECTION("Returns expected weights norm gradient when other regularizations vanish")
   {
      const int n_features = 10;
      const int n_samples = 200;
      const int n_components = 6;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({4, 10, 20});

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);

      const double test_epsilon_states = 1.;
      const double test_epsilon_weights = 0;
      const double test_eta_weights = 0.2;
      const double test_epsilon_parameters = 2;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.weights_gradient(X, S, Gamma, jac_Gamma);

      Eigen::MatrixXd expected_gradient = test_eta_weights * Gamma;

      CHECK(is_equal(jac_Gamma, expected_gradient, tolerance));
   }

   SECTION("Returns expected weights gradient when other regularizations vanish")
   {
      const int n_features = 11;
      const int n_samples = 10;
      const int n_components = 3;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({2, 4});
      const int n_lags = lag_set.size();
      int max_lag = -1;
      for (auto l : lag_set) {
         if (l > max_lag) {
            max_lag = l;
         }
      }

      const Eigen::MatrixXd parameters(Eigen::MatrixXd::Random(n_components, n_lags));

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);
      host_instance.set_parameter_values(parameters);

      const double test_epsilon_states = 0;
      const double test_epsilon_weights = 1.2;
      const double test_eta_weights = 0;
      const double test_epsilon_parameters = 0.2;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.weights_gradient(X, S, Gamma, jac_Gamma);

      Eigen::MatrixXd expected_gradient(
         Eigen::MatrixXd::Zero(n_components, n_samples));

      for (int i = 0; i < n_components; ++i) {
         const Eigen::VectorXd w = parameters.row(i);
         for (int t = max_lag; t < n_samples; ++t) {
            double r = Gamma(i, t);
            for (int l = 0; l < n_lags; ++l) {
               r -= w(l) * Gamma(i, t - lag_set[l]);
            }

            expected_gradient(i, t) += test_epsilon_weights * r;

            for (int l = 0; l < n_lags; ++l) {
               expected_gradient(i, t - lag_set[l]) -=
                  test_epsilon_weights * w(l) * r;
            }
         }
      }

      CHECK(is_equal(jac_Gamma, expected_gradient, tolerance));
   }

   SECTION("Test analytic weights gradient matches numerical gradient")
   {
      const int n_features = 11;
      const int n_samples = 100;
      const int n_components = 5;
      const double h = 1e-5;
      const double tolerance = 1e-5;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      const Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      const std::vector<int> lag_set({1, 2, 3, 10});
      const int n_lags = lag_set.size();
      int max_lag = -1;
      for (auto l : lag_set) {
         if (l > max_lag) {
            max_lag = l;
         }
      }

      const Eigen::MatrixXd parameters(Eigen::MatrixXd::Random(n_components, n_lags));

      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);
      host_instance.set_parameter_values(parameters);

      const double test_epsilon_states = 0.2;
      const double test_epsilon_weights = 1.2;
      const double test_eta_weights = 0.3;
      const double test_epsilon_parameters = 0.2;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.weights_gradient(X, S, Gamma, jac_Gamma);

      Eigen::MatrixXd numerical_gradient(
         Eigen::MatrixXd::Zero(n_components, n_samples));

      for (int i = 0; i < n_components; ++i) {
         for (int t = 0; t < n_samples; ++t) {
            Eigen::MatrixXd Gammaph(Gamma);
            Gammaph(i, t) += h;
            const double fxph = host_instance.penalty(X, S, Gammaph);

            Eigen::MatrixXd Gammamh(Gamma);
            Gammamh(i, t) -= h;
            const double fxmh = host_instance.penalty(X, S, Gammamh);

            numerical_gradient(i, t) = (fxph - fxmh) / (2 * h);
         }
      }

      CHECK(is_equal(jac_Gamma, numerical_gradient, tolerance));
   }

   SECTION("Exact solution is fixed point of parameters update")
   {
      const int n_features = 11;
      const int n_samples = 70;
      const int n_components = 4;
      const double tolerance = 1e-10;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(n_components, n_samples);

      const std::vector<int> lag_set({1, 2});
      const int n_lags = lag_set.size();
      int max_lag = -1;
      for (auto l : lag_set) {
         if (l > max_lag) {
            max_lag = l;
         }
      }
      const Eigen::MatrixXd parameters(Eigen::MatrixXd::Random(n_components, n_lags));

      Gamma.block(0, 0, n_components, max_lag) =
         Eigen::MatrixXd::Random(n_components, max_lag);

      for (int t = max_lag; t < n_samples; ++t) {
         for (int i = 0; i < n_components; ++i) {
            double value = 0;
            for (int l = 0; l < n_lags; ++l) {
               value += parameters(i, l) * Gamma(i, t - lag_set[l]);
            }
            Gamma(i, t) = value;
         }
      }

      std::cout << "Gamma = " << Gamma << '\n';
      GPNH_DAR_eigen_host host_instance;
      host_instance.initialize(lag_set, Gamma);
      host_instance.set_parameter_values(parameters);

      const double test_epsilon_states = 0.2;
      const double test_epsilon_weights = 1.2;
      const double test_eta_weights = 0.3;
      const double test_epsilon_parameters = 0;

      host_instance.set_epsilon_states(test_epsilon_states);
      host_instance.set_epsilon_weights(test_epsilon_weights);
      host_instance.set_eta_weights(test_eta_weights);
      host_instance.set_epsilon_parameters(test_epsilon_parameters);

      host_instance.update_parameters(X, S, Gamma);
      const Eigen::MatrixXd updated_parameters(host_instance.get_parameters());

      std::cout << "parameters = " << parameters << '\n';
      std::cout << "updated = " << updated_parameters << '\n';

      CHECK(is_equal(parameters, updated_parameters, tolerance));
   }
}

#endif
