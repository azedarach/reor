#include "catch.hpp"

#include "reor/kernel_aa.hpp"
#include "reor/numerics_helpers.hpp"
#include "reor/random_matrix.hpp"

#include "comparison_helpers.hpp"

#include <algorithm>
#include <random>

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

TEST_CASE("Test dictionary update with Eigen matrices",
          "[kernel_aa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;

   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("Single dictionary update reduces cost function with delta = 0")
   {
      const int n_features = 10;
      const int n_components = 5;
      const int n_samples = 400;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);
      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      CHECK(final_cost < initial_cost);

      const auto updated_C = aa.get_dictionary();
      CHECK(is_left_stochastic_matrix(C, 1e-12));
   }

   SECTION("Single dictionary update reduces cost function with non-zero delta")
   {
      const int n_features = 10;
      const int n_components = 5;
      const int n_samples = 400;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);
      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 1.2;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      CHECK(final_cost < initial_cost);
   }

   SECTION("Exact solution with delta = 0 is fixed point of update")
   {
      const int n_features = 10;
      const int n_components = 6;
      const int n_samples = 100;
      const double tolerance = 1e-12;

      const Eigen::MatrixXd basis(
         Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      std::vector<int> archetype_indices(n_components);
      std::uniform_int_distribution<> dist(0, n_samples - 1);
      for (int i = 0; i < n_components; ++i) {
         bool new_index = false;
         int current_index = 0;
         while (!new_index) {
            new_index = true;
            current_index = dist(generator);
            for (auto idx : archetype_indices) {
               if (current_index == idx) {
                  new_index = false;
               }
            }
         }
         archetype_indices[i] = current_index;
      }

      Eigen::MatrixXd C(Eigen::MatrixXd::Zero(n_samples, n_components));
      int component = 0;
      for (auto idx : archetype_indices) {
         C(idx, component) = 1;
         for (int i = 0; i < n_components; ++i) {
            if (i == component) {
               S(i, idx) = 1;
            } else {
               S(i, idx) = 0;
            }
         }
         ++component;
      }

      const Eigen::MatrixXd X(basis * S);
      const Eigen::MatrixXd basis_projection = X * C;

      REQUIRE(is_equal(basis, basis_projection, tolerance));
      REQUIRE((X - X * C * S).norm() < tolerance);

      const Eigen::MatrixXd K(X.transpose() * X);

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_dictionary();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      const Eigen::MatrixXd updated_C(aa.get_dictionary());

      CHECK(is_equal(initial_cost, final_cost, tolerance));
      CHECK(is_left_stochastic_matrix(updated_C, tolerance));
      CHECK(is_equal(C, updated_C, tolerance));
   }

   SECTION("Repeated updates converge to fixed point with delta = 0")
   {
      const int n_features = 20;
      const int n_components = 15;
      const int n_samples = 600;
      const int max_iter = 500;
      const double tolerance = 1e-6;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         const int error = aa.update_dictionary();
         REQUIRE(error == 0);
         new_cost = aa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const auto updated_C = aa.get_dictionary();
      CHECK(is_left_stochastic_matrix(updated_C, 1e-12));
   }

   SECTION("Repeated updates converge to fixed point with non-zero delta")
   {
      const int n_features = 30;
      const int n_components = 11;
      const int n_samples = 320;
      const int max_iter = 1000;
      const double tolerance = 1e-4;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 3.2;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         const int error = aa.update_dictionary();
         REQUIRE(error == 0);
         new_cost = aa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const auto updated_C = aa.get_dictionary();
      CHECK(is_left_stochastic_matrix(updated_C, 1e-12));

      const auto updated_alpha = aa.get_scale_factors();
      for (int i = 0; i < n_components; ++i) {
         CHECK(updated_alpha(i, i) >= 1 - delta);
         CHECK(updated_alpha(i, i) <= 1 + delta);
      }
   }
}

TEST_CASE("Test weights update with Eigen matrices",
          "[kernel_aa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;

   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("Single weights update reduces cost function with delta = 0")
   {
      const int n_features = 13;
      const int n_components = 7;
      const int n_samples = 100;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);
      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_weights();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      CHECK(final_cost < initial_cost);
   }

   SECTION("Single weights update reduces cost function with non-zero delta")
   {
      const int n_features = 50;
      const int n_components = 5;
      const int n_samples = 400;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);
      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 1.2;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_weights();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      CHECK(final_cost < initial_cost);
   }

   SECTION("Exact solution with delta = 0 is fixed point of update")
   {
      const int n_features = 30;
      const int n_components = 10;
      const int n_samples = 130;
      const double tolerance = 1e-12;

      const Eigen::MatrixXd basis(
         Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      std::vector<int> archetype_indices(n_components);
      std::uniform_int_distribution<> dist(0, n_samples - 1);
      for (int i = 0; i < n_components; ++i) {
         bool new_index = false;
         int current_index = 0;
         while (!new_index) {
            new_index = true;
            current_index = dist(generator);
            for (auto idx : archetype_indices) {
               if (current_index == idx) {
                  new_index = false;
               }
            }
         }
         archetype_indices[i] = current_index;
      }

      Eigen::MatrixXd C(Eigen::MatrixXd::Zero(n_samples, n_components));
      int component = 0;
      for (auto idx : archetype_indices) {
         C(idx, component) = 1;
         for (int i = 0; i < n_components; ++i) {
            if (i == component) {
               S(i, idx) = 1;
            } else {
               S(i, idx) = 0;
            }
         }
         ++component;
      }

      const Eigen::MatrixXd X(basis * S);

      const Eigen::MatrixXd basis_projection = X * C;
      REQUIRE(is_equal(basis, basis_projection, tolerance));
      REQUIRE((X - X * C * S).norm() < tolerance);

      const Eigen::MatrixXd K(X.transpose() * X);

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      const double initial_cost = aa.cost();

      const int error = aa.update_weights();

      REQUIRE(error == 0);

      const double final_cost = aa.cost();

      const Eigen::MatrixXd updated_S(aa.get_weights());

      CHECK(is_equal(initial_cost, final_cost, tolerance));
      CHECK(is_left_stochastic_matrix(updated_S, tolerance));
      CHECK(is_equal(S, updated_S, tolerance));
   }

   SECTION("Repeated updates converge to fixed point with delta = 0")
   {
      const int n_features = 10;
      const int n_components = 3;
      const int n_samples = 600;
      const int max_iter = 100;
      const double tolerance = 1e-6;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;

      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         const int error = aa.update_weights();
         REQUIRE(error == 0);
         new_cost = aa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const auto updated_S = aa.get_weights();
      CHECK(is_left_stochastic_matrix(updated_S, 1e-12));
   }

   SECTION("Repeated updates converge to fixed point with non-zero delta")
   {
      const int n_features = 30;
      const int n_components = 11;
      const int n_samples = 320;
      const int max_iter = 100;
      const double tolerance = 1e-6;

      const Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples));
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0.3;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;
         const int error = aa.update_weights();
         REQUIRE(error == 0);
         new_cost = aa.cost();

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const auto updated_S = aa.get_weights();
      CHECK(is_left_stochastic_matrix(updated_S, 1e-12));
   }
}

TEST_CASE("Test solution for kernel AA with Eigen matrices",
          "[kernel_aa][eigen_backend]")
{
   using Backend = backends::Eigen_backend<double>;

   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("Finds elements of 3 point convex hull as archetypes for 2D example")
   {
      const int n_features = 2;
      const int n_samples = 50;
      const int n_components = 3;
      const int max_iter = 500;
      const double tolerance = 1e-6;

      Eigen::MatrixXd basis(n_features, n_components);
      basis << 0, 1, 0,
         0, 0, 1;

      Eigen::MatrixXd expected_S(n_components, n_samples);
      random_left_stochastic_matrix(expected_S, generator);

      const std::vector<int> assignments({5, 27, 32});
      for (int i = 0; i < n_components; ++i) {
         expected_S.col(assignments[i]) = Eigen::VectorXd::Zero(n_components);
         expected_S(i, assignments[i]) = 1;
      }

      const Eigen::MatrixXd X(basis * expected_S);
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;
      int error = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;

         error = aa.update_dictionary();
         REQUIRE(error == 0);

         new_cost = aa.cost();
         CHECK(new_cost <= old_cost);

         error = aa.update_weights();
         REQUIRE(error == 0);

         new_cost = aa.cost();
         CHECK(new_cost <= old_cost);

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const Eigen::MatrixXd solution_C(aa.get_dictionary());
      const Eigen::MatrixXd solution_S(aa.get_weights());

      CHECK(is_left_stochastic_matrix(solution_C, 1e-12));
      CHECK(is_left_stochastic_matrix(solution_S, 1e-12));

      std::vector<int> main_components(n_components);
      for (int i = 0; i < n_components; ++i) {
         solution_C.col(i).maxCoeff(&main_components[i]);
      }
      std::sort(std::begin(main_components), std::end(main_components));

      for (int i = 0; i < n_components; ++i) {
         CHECK(main_components[i] == assignments[i]);
      }
   }

   SECTION("Finds elements of 8 point convex hull as archetypes for 3D example")
   {
      const int n_features = 3;
      const int n_samples = 100;
      const int n_components = 8;
      const int max_iter = 500;
      const double tolerance = 1e-6;

      Eigen::MatrixXd basis(n_features, n_components);
      basis << 0, 1, 1, 0, 0, 1, 1, 0,
         0, 0, 1, 1, 0, 0, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1;

      Eigen::MatrixXd expected_S(n_components, n_samples);
      random_left_stochastic_matrix(expected_S, generator);

      const std::vector<int> assignments({8, 9, 15, 20, 34, 56, 78, 90});
      for (int i = 0; i < n_components; ++i) {
         expected_S.col(assignments[i]) = Eigen::VectorXd::Zero(n_components);
         expected_S(i, assignments[i]) = 1;
      }

      const Eigen::MatrixXd X(basis * expected_S);
      const Eigen::MatrixXd K(X.transpose() * X);

      Eigen::MatrixXd C(n_samples, n_components);
      Eigen::MatrixXd S(n_components, n_samples);

      random_left_stochastic_matrix(C, generator);
      random_left_stochastic_matrix(S, generator);

      REQUIRE(is_left_stochastic_matrix(C, 1e-12));
      REQUIRE(is_left_stochastic_matrix(S, 1e-12));

      const double delta = 0;
      KernelAA<Backend> aa(K, C, S, delta);

      double cost_delta = std::numeric_limits<double>::max();
      double old_cost = aa.cost();
      double new_cost = old_cost;
      int iter = 0;
      int error = 0;
      while (std::abs(cost_delta) > tolerance && iter < max_iter) {
         old_cost = new_cost;

         error = aa.update_dictionary();
         REQUIRE(error == 0);

         new_cost = aa.cost();
         CHECK(new_cost <= old_cost);

         error = aa.update_weights();
         REQUIRE(error == 0);

         new_cost = aa.cost();
         CHECK(new_cost <= old_cost);

         cost_delta = new_cost - old_cost;

         CHECK(cost_delta <= 0);

         ++iter;
      }

      REQUIRE(iter < max_iter);

      const Eigen::MatrixXd solution_C(aa.get_dictionary());
      const Eigen::MatrixXd solution_S(aa.get_weights());

      CHECK(is_left_stochastic_matrix(solution_C, 1e-12));
      CHECK(is_left_stochastic_matrix(solution_S, 1e-12));

      std::vector<int> main_components(n_components);
      for (int i = 0; i < n_components; ++i) {
         solution_C.col(i).maxCoeff(&main_components[i]);
      }
      std::sort(std::begin(main_components), std::end(main_components));

      for (int i = 0; i < n_components; ++i) {
         CHECK(main_components[i] == assignments[i]);
      }
   }
}

#endif
