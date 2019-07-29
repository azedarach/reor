#include "catch.hpp"
#include "comparison_helpers.hpp"

#include "reor/l2_spa_regularizations.hpp"
#include "reor/numerics_helpers.hpp"

using namespace reor;

#ifdef HAVE_EIGEN

#include "reor/backends/eigen_backend.hpp"

#include <Eigen/Core>


namespace {

struct GPNH_eigen_host :
      public L2_SPA_GPNH_regularization<backends::Eigen_backend<double> > {};

} // anonymous namespace

TEST_CASE("Test GPNH regularization with Eigen matrices",
          "[l2_spa][l2_spa_regularizations][eigen_backend]")
{
   SECTION("Throws when given negative regularization parameter")
   {
      GPNH_eigen_host host_instance;
      CHECK_THROWS(host_instance.set_epsilon_states(-1));
   }

   SECTION("Returns zero when only one component")
   {
      const int n_features = 15;
      const int n_samples = 700;
      const int n_components = 1;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      GPNH_eigen_host host_instance;
      const double test_epsilon_states = 10.;
      host_instance.set_epsilon_states(test_epsilon_states);

      const double penalty = host_instance.penalty(X, S, Gamma);
      const double expected_penalty = 0;

      CHECK(is_equal(penalty, expected_penalty));
   }

   SECTION("Returns zero jacobian when only one component")
   {
      const int n_features = 20;
      const int n_samples = 300;
      const int n_components = 1;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_S(Eigen::MatrixXd::Random(n_features, n_components));

      REQUIRE(!is_equal(jac_S,
                        Eigen::MatrixXd::Zero(n_features, n_components)));

      GPNH_eigen_host host_instance;
      const double test_epsilon_states = 1.;
      host_instance.set_epsilon_states(test_epsilon_states);

      host_instance.dictionary_gradient(X, S, Gamma, jac_S);

      CHECK(is_equal(jac_S, Eigen::MatrixXd::Zero(n_features, n_components)));
   }

   SECTION("Returns zero affiliations jacobian for arbitrary matrices")
   {
      const int n_features = 12;
      const int n_samples = 50;
      const int n_components = 17;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_Gamma(
         Eigen::MatrixXd::Random(n_components, n_samples));

      REQUIRE(!is_equal(jac_Gamma,
                        Eigen::MatrixXd::Zero(n_components, n_samples)));

            GPNH_eigen_host host_instance;
      const double test_epsilon_states = 1.;
      host_instance.set_epsilon_states(test_epsilon_states);

      host_instance.affiliations_gradient(X, S, Gamma, jac_Gamma);

      CHECK(is_equal(jac_Gamma,
                     Eigen::MatrixXd::Zero(n_components, n_samples)));
   }

   SECTION("Returns zero penalty when regularization parameter is zero")
   {
      const int n_features = 50;
      const int n_samples = 100;
      const int n_components = 5;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Random(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      GPNH_eigen_host host_instance;
      const double test_epsilon_states = 0.;
      host_instance.set_epsilon_states(test_epsilon_states);

      const double penalty = host_instance.penalty(X, S, Gamma);
      const double expected_penalty = 0;

      CHECK(is_equal(penalty, expected_penalty));
   }

   SECTION("Returns expected penalty for small system")
   {
      const int n_features = 2;
      const int n_samples = 432;
      const int n_components = 2;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Zero(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));

      S << 1., -3.,
         2., 2.;

      GPNH_eigen_host host_instance;
      const double test_epsilon_states = 2.;
      host_instance.set_epsilon_states(test_epsilon_states);

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

      CHECK(is_equal(penalty, expected_penalty));
   }

   SECTION("Returns expected dictionary gradient for small system")
   {
      const int n_features = 2;
      const int n_samples = 211;
      const int n_components = 2;

      Eigen::MatrixXd X(Eigen::MatrixXd::Random(n_features, n_samples));
      Eigen::MatrixXd S(Eigen::MatrixXd::Zero(n_features, n_components));
      Eigen::MatrixXd Gamma(Eigen::MatrixXd::Random(n_components, n_samples));
      Eigen::MatrixXd jac_S(Eigen::MatrixXd::Zero(n_features, n_components));

      S << -3., -3.,
         10., 0.;

      GPNH_eigen_host host_instance;
      const double test_epsilon_states = 3.;
      host_instance.set_epsilon_states(test_epsilon_states);

      host_instance.dictionary_gradient(X, S, Gamma, jac_S);

      Eigen::MatrixXd expected_jac_S(
         Eigen::MatrixXd::Zero(n_features, n_components));

      expected_jac_S = n_components * S
         - S * Eigen::MatrixXd::Ones(n_components, n_components);
      expected_jac_S *= 4 * test_epsilon_states /
         (n_features * n_components * (n_components - 1));

      CHECK(is_equal(jac_S, expected_jac_S));
   }
}

#endif
