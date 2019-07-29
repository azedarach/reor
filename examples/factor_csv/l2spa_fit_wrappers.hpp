#ifndef REOR_L2_SPA_FIT_WRAPPERS_HPP_INCLUDED
#define REOR_L2_SPA_FIT_WRAPPERS_HPP_INCLUDED

#include "reor/backend_interface.hpp"
#include "reor/backends/eigen_backend.hpp"
#include "reor/l2_spa.hpp"
#include "reor/matrix_factorization_helpers.hpp"

#include <Eigen/Core>

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace reor {

double calculate_rss(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
double calculate_rmse(const Eigen::MatrixXd&, const Eigen::MatrixXd&);

struct Fit_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd affiliations{};
   double cost{std::numeric_limits<double>::max()};
   double training_rss{std::numeric_limits<double>::max()};
   double training_rmse{std::numeric_limits<double>::max()};
   double test_rss{std::numeric_limits<double>::max()};
   double test_rmse{std::numeric_limits<double>::max()};
   int n_iter{-1};
   double time_seconds{-1};
   bool success{false};
   bool validation_success{false};
};

struct Factorization_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd affiliations{};
   int n_components{1};
   double epsilon_states{0};
   std::size_t n_fits{0};
   std::size_t n_successful_fits{0};
   std::size_t n_successful_validations{0};
   int min_n_iter{-1};
   int max_n_iter{-1};
   double min_cost{-1};
   double max_cost{-1};
   double average_cost{-1};
   double min_training_approx_rmse{-1};
   double max_training_approx_rmse{-1};
   double average_training_approx_rmse{-1};
   double min_training_approx_rss{-1};
   double max_training_approx_rss{-1};
   double average_training_approx_rss{-1};
   double min_test_approx_rmse{-1};
   double max_test_approx_rmse{-1};
   double average_test_approx_rmse{-1};
   double min_test_approx_rss{-1};
   double max_test_approx_rss{-1};
   double average_test_approx_rss{-1};
   double min_time_seconds{-1};
   double max_time_seconds{-1};
   double average_time_seconds{-1};
   bool success{false};
};

template <class Generator>
std::tuple<bool, double, double> validate_l2spa(
   const Eigen::MatrixXd& test_data, const Eigen::MatrixXd& dictionary,
   double epsilon_states, double tolerance, int max_iterations,
   Generator& generator)
{
   using Backend = backends::Eigen_backend<double>;
   using Regularization = L2_SPA_GPNH_regularization<Backend>;

   const int n_samples = test_data.cols();
   const int n_components = dictionary.cols();
   Eigen::MatrixXd initial_affiliations(n_components, n_samples);
   random_left_stochastic_matrix(initial_affiliations, generator);

   L2_SPA<Backend, Regularization> spa(test_data, dictionary,
                                       initial_affiliations);
   spa.set_epsilon_states(epsilon_states);

   std::tuple<bool, int, double> result =
      iterate_factors_until_delta_cost_converged(
         spa, tolerance, max_iterations, false, true);

   const bool success = std::get<0>(result);

   double test_rss = std::numeric_limits<double>::max();
   double test_rmse = std::numeric_limits<double>::max();
   if (success) {
      const Eigen::MatrixXd reconstruction =
         spa.get_dictionary() * spa.get_affiliations();

      test_rss = calculate_rss(test_data, reconstruction);
      test_rmse = calculate_rmse(test_data, reconstruction);
   }

   return std::make_tuple(success, test_rss, test_rmse);
}

template <class Generator>
Fit_result run_l2spa(
   const Eigen::MatrixXd& data, const std::vector<int>& test_set,
   int n_components, double epsilon_states, int n_init,
   Eigen::MatrixXd* initial_dictionary, Eigen::MatrixXd* initial_affiliations,
   double tolerance, int max_iterations, Generator& generator)
{
   using Backend = backends::Eigen_backend<double>;
   using Regularization = L2_SPA_GPNH_regularization<Backend>;

   Fit_result best_result;

   const int n_features = data.rows();
   const int n_samples = data.cols();
   const int n_training_samples = n_samples - test_set.size();

   Eigen::MatrixXd training_data(
      Eigen::MatrixXd::Zero(n_features, n_training_samples));
   std::vector<int> training_indices;
   for (int i = 0; i < n_samples; ++i) {
      const bool in_test_set = std::find(
         std::begin(test_set), std::end(test_set), i) != std::end(test_set);
      if (!in_test_set) {
         training_indices.push_back(i);
      }
   }

   for (int i = 0; i < n_training_samples; ++i) {
      training_data.col(i) = data.col(training_indices[i]);
   }

   Eigen::MatrixXd dictionary(n_features, n_components);
   Eigen::MatrixXd affiliations(n_components, n_training_samples);

   for (int i = 0; i < n_init; ++i) {
      const auto start_time = std::chrono::high_resolution_clock::now();

      if (i == 0 && initial_dictionary) {
         dictionary = *initial_dictionary;
      } else {
         dictionary = Eigen::MatrixXd::Random(n_features, n_components);
      }

      if (i == 0 && initial_affiliations) {
         for (int j = 0; j < n_training_samples; ++j) {
            affiliations.col(j) = initial_affiliations->col(
               training_indices[j]);
         }
      } else {
         random_left_stochastic_matrix(affiliations, generator);
      }

      L2_SPA<Backend, Regularization> spa(
         training_data, dictionary, affiliations);
      spa.set_epsilon_states(epsilon_states);

      const std::tuple<bool, int, double> iteration_result =
         iterate_factors_until_delta_cost_converged(
            spa, tolerance, max_iterations, true, true);

      const bool success = std::get<0>(iteration_result);
      const int n_iter = std::get<1>(iteration_result);
      const double cost = std::get<2>(iteration_result);

      const auto end_time = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> total_time = end_time - start_time;

      if (success && cost < best_result.cost) {
         best_result.dictionary = spa.get_dictionary();
         best_result.affiliations = spa.get_affiliations();
         best_result.cost = cost;
         best_result.n_iter = n_iter;
         best_result.time_seconds = total_time.count();
         best_result.success = success;
      }
   }

   if (best_result.success) {
      const Eigen::MatrixXd reconstruction =
         best_result.dictionary * best_result.affiliations;
      best_result.training_rmse = calculate_rmse(training_data, reconstruction);
      best_result.training_rss = calculate_rss(training_data, reconstruction);

      if (test_set.size() != 0) {
         const int n_test_samples = test_set.size();
         Eigen::MatrixXd test_data(n_features, n_test_samples);
         for (int i = 0; i < n_test_samples; ++i) {
            test_data.col(i) = data.col(test_set[i]);
         }

         std::tuple<bool, double, double> validation_result =
            validate_l2spa(test_data, best_result.dictionary,
                           epsilon_states, tolerance, max_iterations,
                           generator);

         const bool validation_success = std::get<0>(validation_result);
         const double test_rss = std::get<1>(validation_result);
         const double test_rmse = std::get<2>(validation_result);

         best_result.validation_success = validation_success;

         if (validation_success) {
            best_result.test_rss = test_rss;
            best_result.test_rmse = test_rmse;
         }
      }
   }

   // as a final step, compute affiliations for all elements of
   // original data set
   if (best_result.success) {
      Eigen::MatrixXd full_dataset_affiliations(n_components, n_samples);
      random_left_stochastic_matrix(full_dataset_affiliations, generator);

      L2_SPA<Backend, Regularization> spa(
         data, best_result.dictionary, full_dataset_affiliations);
      spa.set_epsilon_states(epsilon_states);

      const auto result = iterate_factors_until_delta_cost_converged(
         spa, tolerance, max_iterations, false, true);
      const bool affiliations_success = std::get<0>(result);
      if (affiliations_success) {
         best_result.affiliations = spa.get_affiliations();
      }
   }

   return best_result;
}

template <class Generator>
Factorization_result run_cross_validated_l2spa(
   const Eigen::MatrixXd& data, const std::vector<std::vector<int> >& test_sets,
   int n_components, double epsilon_states, int n_init,
   double tolerance, int max_iterations,
   Eigen::MatrixXd* initial_dictionary, Eigen::MatrixXd* initial_affiliations,
   Generator& generator, bool verbose)
{
   if (verbose) {
      std::cout << "Running factorization with n_components = " << n_components
                << ", epsilon_states = " << epsilon_states
                << ", n_init = " << n_init << '\n';
   }

   if (test_sets.size() == 0) {
      throw std::runtime_error("no test set specifications given");
   }

   const auto start_time = std::chrono::high_resolution_clock::now();

   std::size_t min_cost_index = 0;
   int min_n_iter = std::numeric_limits<int>::max();
   int max_n_iter = -std::numeric_limits<int>::max();
   double min_cost = std::numeric_limits<double>::max();
   double max_cost = -std::numeric_limits<double>::max();
   double average_cost = 0;
   double min_training_approx_rmse = std::numeric_limits<double>::max();
   double max_training_approx_rmse = -std::numeric_limits<double>::max();
   double average_training_approx_rmse = 0;
   double min_training_approx_rss = std::numeric_limits<double>::max();
   double max_training_approx_rss = -std::numeric_limits<double>::max();
   double average_training_approx_rss = 0;
   double min_test_approx_rmse = std::numeric_limits<double>::max();
   double max_test_approx_rmse = -std::numeric_limits<double>::max();
   double average_test_approx_rmse = 0;
   double min_test_approx_rss = std::numeric_limits<double>::max();
   double max_test_approx_rss = -std::numeric_limits<double>::max();
   double average_test_approx_rss = 0;
   double min_time_seconds = std::numeric_limits<double>::max();
   double max_time_seconds = -std::numeric_limits<double>::max();
   double average_time_seconds = 0;

   std::size_t index = 0;
   std::size_t n_successful_fits = 0;
   std::size_t n_successful_validations = 0;
   std::vector<Fit_result> fit_results;
   for (const auto test_set : test_sets) {
      const auto result = run_l2spa(
         data, test_set, n_components, epsilon_states, n_init,
         initial_dictionary, initial_affiliations,
         tolerance, max_iterations, generator);

      fit_results.push_back(result);

      if (result.success) {
         if (result.cost < min_cost) {
            min_cost = result.cost;
            min_cost_index = index;
         }

         if (result.cost > max_cost) {
            max_cost = result.cost;
         }

         average_cost = (result.cost + n_successful_fits * average_cost) /
            (n_successful_fits + 1);

         if (result.n_iter < min_n_iter) {
            min_n_iter = result.n_iter;
         }

         if (result.n_iter > max_n_iter) {
            max_n_iter = result.n_iter;
         }

         if (result.training_rmse < min_training_approx_rmse) {
            min_training_approx_rmse = result.training_rmse;
         }

         if (result.training_rmse > max_training_approx_rmse) {
            max_training_approx_rmse = result.training_rmse;
         }

         average_training_approx_rmse =
            (result.training_rmse +
             n_successful_fits * average_training_approx_rmse) /
            (n_successful_fits + 1);

         if (result.training_rss < min_training_approx_rss) {
            min_training_approx_rss = result.training_rss;
         }

         if (result.training_rss > max_training_approx_rss) {
            max_training_approx_rss = result.training_rss;
         }

         average_training_approx_rss = (result.training_rss +
                                        n_successful_fits * average_training_approx_rss) /
            (n_successful_fits + 1);

         if (test_set.size() != 0 && result.validation_success) {
            if (result.test_rmse < min_test_approx_rmse) {
               min_test_approx_rmse = result.test_rmse;
            }

            if (result.test_rmse > max_test_approx_rmse) {
               max_test_approx_rmse = result.test_rmse;
            }

            average_test_approx_rmse =
               (result.test_rmse +
                n_successful_fits * average_test_approx_rmse) /
               (n_successful_fits + 1);

            if (result.test_rss < min_test_approx_rss) {
               min_test_approx_rss = result.test_rss;
            }

            if (result.test_rss > max_test_approx_rss) {
               max_test_approx_rss = result.test_rss;
            }

            average_test_approx_rss =
               (result.test_rss +
                n_successful_fits * average_test_approx_rss) /
               (n_successful_fits + 1);

            ++n_successful_validations;
         }

         if (result.time_seconds < min_time_seconds) {
            min_time_seconds = result.time_seconds;
         }

         if (result.time_seconds > max_time_seconds) {
            max_time_seconds = result.time_seconds;
         }

         average_time_seconds =
            (result.time_seconds +
             n_successful_fits * average_time_seconds) /
            (n_successful_fits + 1);

         ++n_successful_fits;
      }

      ++index;
   }

   Factorization_result result;
   result.dictionary = fit_results[min_cost_index].dictionary;
   result.affiliations = fit_results[min_cost_index].affiliations;
   result.n_components = n_components;
   result.epsilon_states = epsilon_states;
   result.n_fits = index;
   result.n_successful_fits = n_successful_fits;
   result.n_successful_validations = n_successful_validations;
   result.min_n_iter = min_n_iter;
   result.max_n_iter = max_n_iter;
   result.min_cost = min_cost;
   result.max_cost = max_cost;
   result.average_cost = average_cost;
   result.min_training_approx_rmse = min_training_approx_rmse;
   result.max_training_approx_rmse = max_training_approx_rmse;
   result.average_training_approx_rmse = average_training_approx_rmse;
   result.min_training_approx_rss = min_training_approx_rss;
   result.max_training_approx_rss = max_training_approx_rss;
   result.average_training_approx_rss = average_training_approx_rss;
   result.min_test_approx_rmse = min_test_approx_rmse;
   result.max_test_approx_rmse = max_test_approx_rmse;
   result.average_test_approx_rmse = average_test_approx_rmse;
   result.min_test_approx_rss = min_test_approx_rss;
   result.max_test_approx_rss = max_test_approx_rss;
   result.average_test_approx_rss = average_test_approx_rss;
   result.min_time_seconds = min_time_seconds;
   result.max_time_seconds = max_time_seconds;
   result.average_time_seconds = average_time_seconds;
   result.success = n_successful_fits != 0;

   const auto end_time = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double> total_time_seconds = end_time - start_time;

   if (verbose) {
      std::cout << "Time required: " << total_time_seconds.count() << "s\n";
   }

   return result;
}

} // namespace reor

#endif
