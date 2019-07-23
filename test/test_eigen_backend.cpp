#include "catch.hpp"

#include <Eigen/Core>

#include "reor/backends/eigen_backend.hpp"
#include "reor/numerics_helpers.hpp"

#include "comparison_helpers.hpp"

using namespace reor;

TEST_CASE("Row and column backend helpers return correct values",
          "[eigen_backend]")
{
   SECTION("Returns correct rows and columns for dynamic sized matrices")
   {
      Eigen::MatrixXd x1(2, 2);
      Eigen::MatrixXd x2(Eigen::MatrixXd::Zero(4, 5));

      CHECK(backends::rows(x1) == x1.rows());
      CHECK(backends::cols(x1) == x1.cols());
      CHECK(backends::rows(x2) == x2.rows());
      CHECK(backends::cols(x2) == x2.cols());
   }

   SECTION("Returns correct rows and columns for fixed size matrices")
   {
      Eigen::Matrix<double, 3, 6> x1;
      Eigen::Matrix<int, 2, 1> x2;

      CHECK(backends::rows(x1) == x1.rows());
      CHECK(backends::cols(x1) == x1.cols());
      CHECK(backends::rows(x2) == x2.rows());
      CHECK(backends::cols(x2) == x2.cols());
   }
}

TEST_CASE("Residual matrix correctly computed",
          "[eigen_backend]")
{
   SECTION("Returns correct residual for dynamic sized matrices")
   {
      Eigen::MatrixXd C(Eigen::MatrixXd::Random(4, 4));
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(4, 3));
      Eigen::MatrixXd B(Eigen::MatrixXd::Random(3, 4));
      Eigen::MatrixXd r1(4, 4);

      backends::matrix_residual(C, A, B, r1);

      Eigen::MatrixXd expected = C - A * B;

      CHECK(is_equal(r1, expected));

      A = Eigen::MatrixXd::Random(10, 10);
      B = Eigen::MatrixXd::Random(12, 10);
      Eigen::MatrixXd r2(10, 12);

      backends::matrix_residual(A * B, A, B, r2);

      CHECK(is_equal(r2, Eigen::MatrixXd::Zero(10, 12)));
   }

   SECTION("Returns correct residual for fixed size matrices")
   {
      Eigen::Matrix<double, 3, 3> C(Eigen::MatrixXd::Random(3, 3));
      Eigen::Matrix<double, 3, 5> A(Eigen::MatrixXd::Random(3, 5));
      Eigen::Matrix<double, 5, 3> B(Eigen::MatrixXd::Random(5, 3));
      Eigen::Matrix<double, 3, 3> r1;

      backends::matrix_residual(C, A, B, r1);
      Eigen::Matrix<double, 3, 3> exp1 = C - A * B;

      CHECK(is_equal(r1, exp1));
   }
}

TEST_CASE("Residual matrix norm correctly computed",
          "[eigen_backend]")
{
   SECTION("Returns correct norm for dynamic size matrices")
   {
      Eigen::MatrixXd C(Eigen::MatrixXd::Random(6, 6));
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(6, 6));
      Eigen::MatrixXd B(Eigen::MatrixXd::Random(6, 6));

      const auto rn1 = backends::matrix_residual_fro_norm(
         C, A, B);
      const double exp1 = (C - A * B).norm();

      CHECK(is_equal(rn1, exp1));
   }

   SECTION("Returns correct norm for fixed size matrices")
   {
      Eigen::Matrix<double, 4, 4> C(Eigen::MatrixXd::Random(4, 4));
      Eigen::Matrix<double, 4, 5> A(Eigen::MatrixXd::Random(4, 5));
      Eigen::Matrix<double, 5, 4> B(Eigen::MatrixXd::Random(5, 4));

      const auto rn1 = backends::matrix_residual_fro_norm(
         C, A, B);
      const double exp1 = (C - A * B).norm();

      CHECK(is_equal(rn1, exp1));
   }
}

TEST_CASE("Test simplex projection for Eigen vectors",
          "[eigen_backend]")
{
   SECTION("Returns correct projection for 1D vector")
   {
      Eigen::VectorXd x(1);
      x << -0.5;

      Eigen::VectorXd expected(1);
      expected << 1.;

      backends::simplex_project_vector(x);

      CHECK(is_equal(x, expected));
   }

   SECTION("Does not change 1D vector in simplex")
   {
      Eigen::VectorXd x(1);
      x << 1.;

      Eigen::VectorXd projection(x);

      backends::simplex_project_vector(projection);

      CHECK(is_equal(x, projection));
   }

   SECTION("Returns correct projection for 2D vector")
   {
      Eigen::VectorXd x1(2);
      x1 << 0.8, 0.8;

      Eigen::VectorXd expected_1(2);
      expected_1 << 0.5, 0.5;

      backends::simplex_project_vector(x1);

      CHECK(is_equal(x1, expected_1));

      Eigen::VectorXd x2(2);
      x2 << 0., 2.;

      Eigen::VectorXd expected_2(2);
      expected_2 << 0., 1.;

      backends::simplex_project_vector(x2);

      CHECK(is_equal(x2, expected_2));

      Eigen::VectorXd x3(2);
      x3 << 0.5, -0.5;

      Eigen::VectorXd expected_3(2);
      expected_3 << 1., 0.;

      backends::simplex_project_vector(x3);

      CHECK(is_equal(x3, expected_3));
   }

   SECTION("Does not change 2D vector in simplex")
   {
      Eigen::VectorXd x(2);
      x << 0.4, 0.6;

      Eigen::VectorXd projection(x);

      backends::simplex_project_vector(projection);

      CHECK(is_equal(x, projection));
   }

   SECTION("Projected 5D vector is in simplex")
   {
      const int n_features = 5;
      const double tolerance = 1e-14;

      Eigen::VectorXd x(Eigen::VectorXd::Random(n_features));
      backends::simplex_project_vector(x);

      CHECK((x.array() >= 0).all());

      const double sum = x.sum();

      CHECK(is_equal(sum, 1., tolerance));
   }

   SECTION("Projected 10D vector is in simplex")
   {
      const int n_features = 10;
      const double tolerance = 1e-14;

      Eigen::VectorXd x(Eigen::VectorXd::Random(n_features));
      backends::simplex_project_vector(x);

      CHECK((x.array() >= 0).all());

      const double sum = x.sum();

      CHECK(is_equal(sum, 1., tolerance));
   }

   SECTION("Projected 100D vector is in simplex")
   {
      const int n_features = 100;
      const double tolerance = 1e-14;

      Eigen::VectorXd x(Eigen::VectorXd::Random(n_features));
      backends::simplex_project_vector(x);

      CHECK((x.array() >= 0).all());

      const double sum = x.sum();

      CHECK(is_equal(sum, 1., tolerance));
   }

}

TEST_CASE("Test simplex projection for Eigen matrices",
          "[eigen_backend]")
{
   SECTION("Does not change 1D columns already in simplex")
   {
      const int n_features = 1;
      const int n_samples = 15;

      const Eigen::MatrixXd X(Eigen::MatrixXd::Ones(n_features, n_samples));

      Eigen::MatrixXd projection(X);
      backends::simplex_project_columns(projection);

      CHECK(is_equal(projection, X));
   }

   SECTION("Correctly projects 1D columns")
   {
      const int n_features = 1;
      const int n_samples = 50;
      const double tolerance = 1e-15;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      backends::simplex_project_columns(X);

      const Eigen::MatrixXd expected(
         Eigen::MatrixXd::Ones(n_features, n_samples));

      CHECK(is_equal(X, expected, tolerance));
   }

   SECTION("Does not change 2D columns already in simplex")
   {
      const int n_features = 2;
      const int n_samples = 10;
      const double tolerance = 1e-15;

      Eigen::MatrixXd X(
         Eigen::MatrixXd::Random(n_features, n_samples).cwiseAbs());

      Eigen::RowVectorXd col_sums = X.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         for (int j = 0; j < n_features; ++j) {
            X(j, i) /= col_sums(i);
         }
      }

      for (int i = 0; i < n_samples; ++i) {
         const double sum = X.col(i).sum();
         REQUIRE(is_equal(sum, 1., tolerance));
      }

      Eigen::MatrixXd projection(X);
      backends::simplex_project_columns(projection);

      CHECK(is_equal(projection, X));
   }

   SECTION("Correctly projects 2D columns")
   {
      const int n_features = 2;
      const int n_samples = 3;
      const double tolerance = 1e-15;

      Eigen::MatrixXd X(n_features, n_samples);

      X << 0.5, 0.5, 0,
         0.5, 1., -0.5;

      Eigen::MatrixXd expected(n_features, n_samples);

      expected << 0.5, 0.25, 0.75,
         0.5, 0.75, 0.25;

      backends::simplex_project_columns(X);

      CHECK(is_equal(X, expected, tolerance));
   }

   SECTION("Columns of 5D projection are in simplex")
   {
      const int n_features = 5;
      const int n_samples = 57;
      const double tolerance = 1e-15;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      backends::simplex_project_columns(X);

      CHECK((X.array() >= 0.).all());

      Eigen::RowVectorXd col_sums = X.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         CHECK(is_equal(col_sums(i), 1., tolerance));
      }
   }

   SECTION("Columns of 317D projection are in simplex")
   {
      const int n_features = 317;
      const int n_samples = 341;
      const double tolerance = 1e-14;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      backends::simplex_project_columns(X);

      CHECK((X.array() >= 0.).all());

      Eigen::RowVectorXd col_sums = X.colwise().sum();
      for (int i = 0; i < n_samples; ++i) {
         CHECK(is_equal(col_sums(i), 1., tolerance));
      }
   }
}
