#include "catch.hpp"

#include "reor/l2_spa.hpp"
#include "reor/numerics_helpers.hpp"

#include <limits>

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

TEST_CASE("Test dictionary update with no regularization and Eigen matrices",
          "[l2_spa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;

   SECTION("Single dictionary update reduces cost function")
   {
      const int n_features = 7;
      const int n_components = 5;
      const int n_samples = 450;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(
         Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(
         Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         for (int k = 0; k < n_components; ++k) {
            Gamma(k, i) /= col_sums(i);
         }
      }

      L2_SPA<Backend> spa(X, S, Gamma);

      const double initial_cost = spa.cost();

      const int error = spa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = spa.cost();

      CHECK(final_cost < initial_cost);
   }

   SECTION("Exact solution is fixed point of update")
   {
      const int n_features = 10;
      const int n_components = 6;
      const int n_samples = 40;
      const double tolerance = 1e-12;

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

      const double initial_cost = spa.cost();

      const int error = spa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = spa.cost();

      CHECK(is_equal(initial_cost, final_cost, tolerance));

      // @todo check actual matrices
   }

   SECTION("Repeated updates converge to fixed point")
   {
      const int n_features = 13;
      const int n_components = 3;
      const int n_samples = 50;
      const int max_iter = 10;
      const double tolerance = 1e-14;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
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

      L2_SPA<Backend> spa(X, S, Gamma);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = spa.cost();
      double new_cost = old_cost;
      int iter = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         spa.update_dictionary();
         new_cost = spa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);
   }
}

TEST_CASE("Test dictionary update with GPNH regularization and Eigen matrices",
          "[l2_spa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;
   using Regularization = L2_SPA_GPNH_regularization<Backend>;

   SECTION("Single dictionary update reduces cost function")
   {
      const int n_features = 7;
      const int n_components = 5;
      const int n_samples = 450;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(
         Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(
         Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         for (int k = 0; k < n_components; ++k) {
            Gamma(k, i) /= col_sums(i);
         }
      }

      L2_SPA<Backend, Regularization> spa(X, S, Gamma);
      const double test_epsilon_states = 3.;
      spa.set_epsilon_states(test_epsilon_states);

      const double initial_cost = spa.cost();

      const int error = spa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = spa.cost();

      CHECK(final_cost < initial_cost);
   }

   SECTION("Exact solution is fixed point of update when no regularization")
   {
      const int n_features = 10;
      const int n_components = 6;
      const int n_samples = 40;
      const double tolerance = 1e-12;

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

      L2_SPA<Backend, Regularization> spa(X, S, Gamma);
      const double test_epsilon_states = 0.;
      spa.set_epsilon_states(test_epsilon_states);

      const double initial_cost = spa.cost();

      const int error = spa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = spa.cost();

      CHECK(is_equal(initial_cost, final_cost, tolerance));

      // @todo check actual matrices
   }

   SECTION("Repeated updates converge to fixed point")
   {
      const int n_features = 13;
      const int n_components = 3;
      const int n_samples = 50;
      const int max_iter = 10;
      const double tolerance = 1e-14;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
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

      L2_SPA<Backend, Regularization> spa(X, S, Gamma);
      const double test_epsilon_states = 1.;
      spa.set_epsilon_states(test_epsilon_states);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = spa.cost();
      double new_cost = old_cost;
      int iter = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         spa.update_dictionary();
         new_cost = spa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);
   }
}

#endif
