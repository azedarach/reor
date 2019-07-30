#ifndef REOR_L2_TRSPA_FIT_WRAPPERS_HPP_INCLUDED
#define REOR_L2_TRSPA_FIT_WRAPPERS_HPP_INCLUDED

#include "reor/backend_interface.hpp"
#include "reor/backends/eigen_backend.hpp"
#include "reor/gpnh_dar_regularizer.hpp"
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

double calculate_rss(const Eigen::MatrixXd&);
double calculate_rss(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
double calculate_rmse(const Eigen::MatrixXd&);
double calculate_rmse(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
Eigen::MatrixXd get_predicted_weights(
   const Eigen::MatrixXd&,
   const std::vector<int>&, const Eigen::MatrixXd&,
   int);
Eigen::MatrixXd calculate_prediction_errors(
   const Eigen::MatrixXd&,
   const Eigen::MatrixXd&,
   const std::vector<int>&,
   const Eigen::MatrixXd&,
   const Eigen::MatrixXd&);

struct Validation_result {
   double approx_rss{std::numeric_limits<double>::max()};
   double approx_rmse{std::numeric_limits<double>::max()};
   double pred_rss{std::numeric_limits<double>::max()};
   double pred_rmse{std::numeric_limits<double>::max()};
   bool success{false};
};

struct Fit_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd weights{};
   Eigen::MatrixXd parameters{};
   double cost{std::numeric_limits<double>::max()};
   double training_approx_rss{std::numeric_limits<double>::max()};
   double training_approx_rmse{std::numeric_limits<double>::max()};
   double test_approx_rss{std::numeric_limits<double>::max()};
   double test_approx_rmse{std::numeric_limits<double>::max()};
   double test_pred_rss{std::numeric_limits<double>::max()};
   double test_pred_rmse{std::numeric_limits<double>::max()};
   int n_iter{-1};
   double time_seconds{-1};
   bool success{false};
   bool validation_success{false};
};

struct Factorization_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd weights{};
   Eigen::MatrixXd parameters{};
   int n_components{1};
   double epsilon_states{0};
   double epsilon_weights{0};
   double eta_weights{0};
   double epsilon_parameters{0};
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
   double min_test_pred_rss{-1};
   double max_test_pred_rss{-1};
   double average_test_pred_rss{-1};
   double min_test_pred_rmse{-1};
   double max_test_pred_rmse{-1};
   double average_test_pred_rmse{-1};
   double min_time_seconds{-1};
   double max_time_seconds{-1};
   double average_time_seconds{-1};
   bool success{false};
};

template <class Generator>
Validation_result validate_l2trspa(
   const Eigen::MatrixXd& test_data,
   const Eigen::MatrixXd& dictionary, const Eigen::MatrixXd& parameters,
   double epsilon_states, double epsilon_weights, double eta_weights,
   double epsilon_parameters, const std::vector<int>& lag_set,
   int horizon, double tolerance, int max_iterations,
   Generator& generator)
{
   using Backend = backends::Eigen_backend<double>;
   using Regularization = GPNH_DAR_regularizer<Backend>;

   const int n_samples = test_data.cols();
   const int n_features = test_data.rows();
   const int n_components = dictionary.cols();

   int max_lag = -1;
   for (auto l : lag_set) {
      if (l > max_lag) {
         max_lag = l;
      }
   }

   const int t_start = max_lag;
   const int t_end = n_samples - horizon;
   const int n_predictions = t_end - t_start;

   bool validation_success = true;

   std::cout << "n_samples = " << n_samples << '\n';
   std::cout << "t_start = " << t_start << '\n';
   Eigen::MatrixXd prediction_errors(
      Eigen::MatrixXd::Zero(n_features, n_predictions));

   Eigen::MatrixXd previous_weights(n_components, max_lag);
   random_left_stochastic_matrix(previous_weights, generator);
   for (int i = 0; i < n_predictions; ++i) {
      const int n_observed = max_lag + i;
      Eigen::MatrixXd observed_data(n_features, n_observed);
      observed_data = test_data.block(0, 0, n_features, n_observed);

      Eigen::MatrixXd future_data(n_features, n_samples - n_observed);
      future_data = test_data.block(0, n_observed, n_features,
                                    n_samples - n_observed);

      Eigen::MatrixXd initial_weights(n_components, n_observed);
      random_left_stochastic_matrix(initial_weights, generator);
      initial_weights.block(0, 0, n_components, n_observed - 1)
         = previous_weights;

      L2_SPA<Backend, Regularization> spa(
         observed_data, dictionary, initial_weights);

      spa.initialize(lag_set, initial_weights);
      spa.set_parameter_values(parameters);

      spa.set_epsilon_states(epsilon_states);
      spa.set_epsilon_weights(epsilon_weights);
      spa.set_eta_weights(eta_weights);
      spa.set_epsilon_parameters(epsilon_parameters);

      std::tuple<bool, int, double> result =
         iterate_factors_until_delta_cost_converged(
            spa, tolerance, max_iterations, false, true, false);

      const bool success = std::get<0>(result);

      if (success) {
         Eigen::MatrixXd observed_weights = spa.get_weights();
         previous_weights = observed_weights;

         Eigen::MatrixXd current_prediction_errors =
            calculate_prediction_errors(
               observed_weights, future_data, lag_set,
               parameters, dictionary);
         prediction_errors.col(i) = current_prediction_errors.col(
            horizon - 1);
      } else {
         validation_success = false;
      }
   }
   std::cout << "prediction_errors = " << prediction_errors << '\n';
   Validation_result result;
   if (validation_success) {
      result.pred_rss = calculate_rss(prediction_errors);
      result.pred_rmse = calculate_rmse(prediction_errors);

      Eigen::MatrixXd initial_weights(n_components, n_samples);
      random_left_stochastic_matrix(initial_weights, generator);

      L2_SPA<Backend, Regularization> spa(
         test_data, dictionary, initial_weights);

      spa.initialize(lag_set, initial_weights);
      spa.set_parameter_values(parameters);

      spa.set_epsilon_states(epsilon_states);
      spa.set_epsilon_weights(epsilon_weights);
      spa.set_eta_weights(eta_weights);
      spa.set_epsilon_parameters(epsilon_parameters);

      std::tuple<bool, int, double> full_result =
         iterate_factors_until_delta_cost_converged(
            spa, tolerance, max_iterations, false, true, false);

      const bool success = std::get<0>(full_result);
      if (success) {
         const Eigen::MatrixXd reconstruction =
            spa.get_dictionary() * spa.get_weights();

         result.approx_rss = calculate_rss(test_data, reconstruction);
         result.approx_rmse = calculate_rmse(test_data, reconstruction);
      } else {
         validation_success = false;
      }
   }

   result.success = validation_success;

   return result;
}

template <class Generator>
Fit_result run_l2trspa(
   const Eigen::MatrixXd& data, double evaluation_fraction,
   int n_components, double epsilon_states,
   double epsilon_weights, double eta_weights, double epsilon_parameters,
   const std::vector<int>& lag_set, int n_init,
   Eigen::MatrixXd* initial_dictionary, Eigen::MatrixXd* initial_weights,
   double tolerance, int max_iterations, Generator& generator)
{
   using Backend = backends::Eigen_backend<double>;
   using Regularization = GPNH_DAR_regularizer<Backend>;

   const int horizon = 1;

   Fit_result best_result;

   const int n_features = data.rows();
   const int n_samples = data.cols();
   const int n_training_samples = std::min(
      static_cast<int>(
         std::floor(
            (1 - evaluation_fraction) * n_samples)),
      n_samples - 1);
   const int n_test_samples = n_samples - n_training_samples;

   int max_lag = -1;
   for (auto l : lag_set) {
      if (l > max_lag) {
         max_lag = l;
      }
   }

   Eigen::MatrixXd training_data(
      Eigen::MatrixXd::Zero(n_features, n_training_samples));
   training_data = data.block(0, 0, n_features, n_training_samples);
   Eigen::MatrixXd test_data(
      Eigen::MatrixXd::Zero(n_features, n_test_samples));
   test_data = data.block(0, n_training_samples, n_features, n_test_samples);

   Eigen::MatrixXd dictionary(n_features, n_components);
   Eigen::MatrixXd weights(n_components, n_training_samples);

   for (int i = 0; i < n_init; ++i) {
      const auto start_time = std::chrono::high_resolution_clock::now();

      if (i == 0 && initial_dictionary) {
         dictionary = *initial_dictionary;
      } else {
         dictionary = Eigen::MatrixXd::Random(n_features, n_components);
      }

      if (i == 0 && initial_weights) {
         weights = initial_weights->block(0, 0, n_components, n_training_samples);;
      } else {
         random_left_stochastic_matrix(weights, generator);
      }

      L2_SPA<Backend, Regularization> spa(
         training_data, dictionary, weights);

      spa.initialize(lag_set, weights);

      spa.set_epsilon_states(epsilon_states);
      spa.set_epsilon_weights(epsilon_weights);
      spa.set_eta_weights(eta_weights);
      spa.set_epsilon_parameters(epsilon_parameters);

      const std::tuple<bool, int, double> iteration_result =
         iterate_factors_until_delta_cost_converged(
            spa, tolerance, max_iterations, true, true, true);

      const bool success = std::get<0>(iteration_result);
      const int n_iter = std::get<1>(iteration_result);
      const double cost = std::get<2>(iteration_result);

      const auto end_time = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> total_time = end_time - start_time;

      if (success && cost < best_result.cost) {
         best_result.dictionary = spa.get_dictionary();
         best_result.weights = spa.get_weights();
         best_result.parameters = spa.get_parameters();
         best_result.cost = cost;
         best_result.n_iter = n_iter;
         best_result.time_seconds = total_time.count();
         best_result.success = success;
      }
   }

   if (best_result.success) {
      const Eigen::MatrixXd reconstruction =
         best_result.dictionary * best_result.weights;
      best_result.training_approx_rmse = calculate_rmse(
         training_data, reconstruction);
      best_result.training_approx_rss = calculate_rss(
         training_data, reconstruction);

      if (n_test_samples >= max_lag + horizon) {
         Validation_result validation_result =
            validate_l2trspa(
               test_data, best_result.dictionary, best_result.parameters,
               epsilon_states, epsilon_weights, eta_weights,
               epsilon_parameters, lag_set,
               horizon, tolerance, max_iterations,
               generator);

         best_result.validation_success = validation_result.success;

         if (best_result.validation_success) {
            best_result.test_approx_rss = validation_result.approx_rss;
            best_result.test_approx_rmse = validation_result.approx_rmse;
            best_result.test_pred_rss = validation_result.pred_rss;
            best_result.test_pred_rmse = validation_result.pred_rmse;
         }
      }
   }

   // as a final step, compute weights for all elements of
   // original data set
   if (best_result.success) {
      Eigen::MatrixXd full_dataset_weights(n_components, n_samples);
      random_left_stochastic_matrix(full_dataset_weights, generator);

      L2_SPA<Backend, Regularization> spa(
         data, best_result.dictionary, full_dataset_weights);

      spa.initialize(lag_set, full_dataset_weights);
      spa.set_parameter_values(best_result.parameters);

      spa.set_epsilon_states(epsilon_states);
      spa.set_epsilon_weights(epsilon_weights);
      spa.set_eta_weights(eta_weights);
      spa.set_epsilon_parameters(epsilon_parameters);

      const auto result = iterate_factors_until_delta_cost_converged(
         spa, tolerance, max_iterations, false, true, false);
      const bool weights_success = std::get<0>(result);
      if (weights_success) {
         best_result.weights = spa.get_weights();
      }
   }

   return best_result;
}

template <class Generator>
Factorization_result run_and_evaluate_l2trspa(
   const Eigen::MatrixXd& data, double evaluation_fraction,
   int n_components, double epsilon_states,
   double epsilon_weights, double eta_weights, double epsilon_parameters,
   const std::vector<int>& lag_set,
   int n_init, double tolerance, int max_iterations,
   Eigen::MatrixXd* initial_dictionary, Eigen::MatrixXd* initial_weights,
   Generator& generator, bool verbose)
{
   if (verbose) {
      std::cout << "Running factorization with n_components = " << n_components
                << ", epsilon_states = " << epsilon_states
                << ", epsilon_weights = " << epsilon_weights
                << ", eta_weights = " << eta_weights
                << ", epsilon_parameters = " << epsilon_parameters
                << ", n_init = " << n_init << '\n';
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
   double min_test_pred_rmse = std::numeric_limits<double>::max();
   double max_test_pred_rmse = -std::numeric_limits<double>::max();
   double average_test_pred_rmse = 0;
   double min_test_pred_rss = std::numeric_limits<double>::max();
   double max_test_pred_rss = -std::numeric_limits<double>::max();
   double average_test_pred_rss = 0;
   double min_time_seconds = std::numeric_limits<double>::max();
   double max_time_seconds = -std::numeric_limits<double>::max();
   double average_time_seconds = 0;

   std::size_t index = 0;
   std::size_t n_successful_fits = 0;
   std::size_t n_successful_validations = 0;
   std::vector<Fit_result> fit_results;

   const auto result = run_l2trspa(
      data, evaluation_fraction,
      n_components, epsilon_states,
      epsilon_weights, eta_weights, epsilon_parameters,
      lag_set, n_init, initial_dictionary, initial_weights,
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

      if (result.training_approx_rmse < min_training_approx_rmse) {
         min_training_approx_rmse = result.training_approx_rmse;
      }

      if (result.training_approx_rmse > max_training_approx_rmse) {
         max_training_approx_rmse = result.training_approx_rmse;
      }

      average_training_approx_rmse =
         (result.training_approx_rmse +
          n_successful_fits * average_training_approx_rmse) /
         (n_successful_fits + 1);

      if (result.training_approx_rss < min_training_approx_rss) {
         min_training_approx_rss = result.training_approx_rss;
      }

      if (result.training_approx_rss > max_training_approx_rss) {
         max_training_approx_rss = result.training_approx_rss;
      }

      average_training_approx_rss = (result.training_approx_rss +
                                     n_successful_fits * average_training_approx_rss) /
         (n_successful_fits + 1);

      if (result.validation_success) {
         if (result.test_approx_rmse < min_test_approx_rmse) {
            min_test_approx_rmse = result.test_approx_rmse;
         }

         if (result.test_approx_rmse > max_test_approx_rmse) {
            max_test_approx_rmse = result.test_approx_rmse;
         }

         average_test_approx_rmse =
            (result.test_approx_rmse +
             n_successful_fits * average_test_approx_rmse) /
            (n_successful_fits + 1);

         if (result.test_approx_rss < min_test_approx_rss) {
            min_test_approx_rss = result.test_approx_rss;
         }

         if (result.test_approx_rss > max_test_approx_rss) {
            max_test_approx_rss = result.test_approx_rss;
         }

         average_test_approx_rss =
            (result.test_approx_rss +
             n_successful_fits * average_test_approx_rss) /
            (n_successful_fits + 1);

         if (result.test_pred_rmse < min_test_pred_rmse) {
            min_test_pred_rmse = result.test_pred_rmse;
         }

         if (result.test_pred_rmse > max_test_pred_rmse) {
            max_test_pred_rmse = result.test_pred_rmse;
         }

         average_test_pred_rmse =
            (result.test_pred_rmse +
             n_successful_fits * average_test_pred_rmse) /
            (n_successful_fits + 1);

         if (result.test_pred_rss < min_test_pred_rss) {
            min_test_pred_rss = result.test_pred_rss;
         }

         if (result.test_pred_rss > max_test_pred_rss) {
            max_test_pred_rss = result.test_pred_rss;
         }

         average_test_pred_rss =
            (result.test_pred_rss +
             n_successful_fits * average_test_pred_rss) /
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
      ++index;
   }

   Factorization_result combined_result;
   combined_result.dictionary = fit_results[min_cost_index].dictionary;
   combined_result.weights = fit_results[min_cost_index].weights;
   combined_result.parameters = fit_results[min_cost_index].parameters;
   combined_result.n_components = n_components;
   combined_result.epsilon_states = epsilon_states;
   combined_result.epsilon_weights = epsilon_weights;
   combined_result.eta_weights = eta_weights;
   combined_result.epsilon_parameters = epsilon_parameters;
   combined_result.n_fits = index;
   combined_result.n_successful_fits = n_successful_fits;
   combined_result.n_successful_validations = n_successful_validations;
   combined_result.min_n_iter = min_n_iter;
   combined_result.max_n_iter = max_n_iter;
   combined_result.min_cost = min_cost;
   combined_result.max_cost = max_cost;
   combined_result.average_cost = average_cost;
   combined_result.min_training_approx_rmse = min_training_approx_rmse;
   combined_result.max_training_approx_rmse = max_training_approx_rmse;
   combined_result.average_training_approx_rmse = average_training_approx_rmse;
   combined_result.min_training_approx_rss = min_training_approx_rss;
   combined_result.max_training_approx_rss = max_training_approx_rss;
   combined_result.average_training_approx_rss = average_training_approx_rss;
   combined_result.min_test_approx_rmse = min_test_approx_rmse;
   combined_result.max_test_approx_rmse = max_test_approx_rmse;
   combined_result.average_test_approx_rmse = average_test_approx_rmse;
   combined_result.min_test_approx_rss = min_test_approx_rss;
   combined_result.max_test_approx_rss = max_test_approx_rss;
   combined_result.average_test_approx_rss = average_test_approx_rss;
   combined_result.min_test_pred_rmse = min_test_pred_rmse;
   combined_result.max_test_pred_rmse = max_test_pred_rmse;
   combined_result.average_test_pred_rmse = average_test_pred_rmse;
   combined_result.min_test_pred_rss = min_test_pred_rss;
   combined_result.max_test_pred_rss = max_test_pred_rss;
   combined_result.average_test_pred_rss = average_test_pred_rss;
   combined_result.min_time_seconds = min_time_seconds;
   combined_result.max_time_seconds = max_time_seconds;
   combined_result.average_time_seconds = average_time_seconds;
   combined_result.success = n_successful_fits != 0;

   const auto end_time = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double> total_time_seconds = end_time - start_time;

   if (verbose) {
      std::cout << "Time required: " << total_time_seconds.count() << "s\n";
   }

   return combined_result;
}

} // namespace reor

#endif
