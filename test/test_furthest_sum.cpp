#include "catch.hpp"

#include "reor/furthest_sum.hpp"
#include "reor/random_matrix.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>

TEST_CASE("Test FurthestSum with Eigen matrices",
          "[furthest_sum][eigen_backend]")
{
   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("Throws error when given non-square dissimilarity matrix")
   {
      const int n_features = 10;
      const int n_samples = 20;
      const int n_components = 2;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));

      CHECK_THROWS(furthest_sum(X, n_components, 0));
   }

   SECTION("Throws error when given out of bounds starting index")
   {
      const int n_samples = 10;
      const int n_components = 5;

      const Eigen::MatrixXd K(Eigen::MatrixXd::Random(n_samples, n_samples));

      CHECK_THROWS(furthest_sum(K, n_components, n_samples + 10));
   }

   SECTION("Throws error when starting index is also excluded")
   {
      const int n_samples = 9;
      const int n_components = 8;

      const Eigen::MatrixXd K(Eigen::MatrixXd::Random(n_samples, n_samples));

      std::vector<int> exclude(n_samples);
      std::iota(std::begin(exclude), std::end(exclude), 0);

      CHECK_THROWS(furthest_sum(K, n_components, 0, exclude));
   }

   SECTION("Throws error when not enough points to fill requested number of components")
   {
      const int n_samples = 32;
      const int n_components = 5;
      const int n_exclude = n_samples - n_components + 2;

      const Eigen::MatrixXd K(
         Eigen::MatrixXd::Random(n_samples, n_samples));

      std::vector<int> exclude(n_exclude);
      std::iota(std::begin(exclude), std::end(exclude), 0);

      REQUIRE(n_components + n_exclude > n_samples);

      CHECK_THROWS(furthest_sum(K, n_components, n_samples - 1,
                                exclude));
   }

   SECTION("Returns empty vector when no components requested")
   {
      const int n_samples = 6;
      const int n_components = 0;

      const Eigen::MatrixXd K(Eigen::MatrixXd::Random(n_samples, n_samples));

      const auto result = furthest_sum(K, n_components, 0);

      CHECK(result.size() == n_components);
   }

   SECTION("Returns all indices when number of components equals number of points")
   {
      const int n_samples = 20;
      const int n_components = n_samples;

      const Eigen::MatrixXd K(
         Eigen::MatrixXd::Random(n_samples, n_samples).cwiseAbs());

      auto result = furthest_sum(K, n_components, 5);
      std::sort(std::begin(result), std::end(result));

      CHECK(result.size() == n_components);

      bool has_duplicates = false;
      std::vector<int> found_indices;
      for (auto idx : result) {
         for (auto found_idx : found_indices) {
            if (idx == found_idx) {
               has_duplicates = true;
               break;
            }
         }
         found_indices.push_back(idx);
      }

      CHECK(!has_duplicates);

      std::vector<int> expected(n_components);
      std::iota(std::begin(expected), std::end(expected), 0);

      for (int i = 0; i < n_components; ++i) {
         CHECK(expected[i] == result[i]);
      }
   }

   SECTION("Returns correct index when only sample present")
   {
      const int n_samples = 1;
      const int n_components = 1;

      const Eigen::MatrixXd K(
         Eigen::MatrixXd::Random(n_samples, n_samples).cwiseAbs());

      const auto result = furthest_sum(K, n_components, 0);

      CHECK(result.size() == 1);
      CHECK(result[0] == 0);
   }

   SECTION("Returns non-excluded index when only possible")
   {
      const int n_samples = 102;
      const int n_components = 1;

      const Eigen::MatrixXd K(
         Eigen::MatrixXd::Random(n_samples, n_samples).cwiseAbs());

      const int leave_in_index = 74;
      std::vector<int> exclude(n_samples - 1);
      int pos = 0;
      for (int i = 0; i < n_samples; ++i) {
         if (i != leave_in_index) {
            exclude[pos] = i;
            pos++;
         }
      }

      const auto result = furthest_sum(K, n_components, leave_in_index,
                                       exclude);

      CHECK(result.size() == 1);
      CHECK(result[0] == leave_in_index);
   }

   SECTION("Selects correct elements out of three for all starting points")
   {
      const int n_samples = 3;
      const int n_components = 2;
      const int max_extra_steps = 10;
      const std::vector<int> exclude;

      Eigen::MatrixXd K(n_samples, n_samples);
      K << 0, 1, 2,
         1, 0, 0.5,
         2, 0.5, 0;

      const std::vector<int> expected({0, 2});
      for (int i = 0; i < n_samples; ++i) {
         for (int x = 1; x <= max_extra_steps; ++x) {
            auto result = furthest_sum(K, n_components, i,
                                       exclude, x);
            std::sort(std::begin(result), std::end(result));
            CHECK(result.size() == n_components);
            for (int j = 0; j < n_components; ++j) {
               CHECK(result[j] == expected[j]);
            }
         }
      }

      SECTION("Selects elements in convex hull of dataset")
      {
         const int n_features = 2;
         const int n_samples = 10;

         Eigen::MatrixXd basis(n_features, 4);
         basis << 0, 1, 1, 0,
            0, 0, 1, 1;
         const int n_basis = basis.cols();

         Eigen::MatrixXd weights(n_basis, n_samples);
         random_left_stochastic_matrix(weights, generator);

         const std::vector<int> assignments({0, 4, 6, 9});
         for (int i = 0; i < n_basis; ++i) {
            weights.col(assignments[i]) = Eigen::VectorXd::Zero(n_basis);
            weights(i, assignments[i]) = 1;
         }

         const Eigen::MatrixXd X(basis * weights);
         Eigen::MatrixXd K(n_samples, n_samples);
         for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_samples; ++j) {
               K(i, j) = (X.col(i) - X.col(j)).norm();
            }
         }

         const int n_components = basis.cols();

         auto result = furthest_sum(K, n_components, 1);
         std::sort(std::begin(result), std::end(result));

         CHECK(result.size() == n_components);

         for (int i = 0; i < n_components; ++i) {
            CHECK(result[i] == assignments[i]);
         }
      }
   }
}

#endif
