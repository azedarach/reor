#ifndef REOR_MATRIX_FACTORIZATION_HELPERS_HPP_INCLUDED
#define REOR_MATRIX_FACTORIZATION_HELPERS_HPP_INCLUDED

/**
 * @file matrix_factorization_helpers.hpp
 * @brief contains definitions of helper and convenience routines
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace reor {

namespace detail {

template <class Factorization>
struct Factorization_delta_cost_iterator {
   double tolerance{1e-8};
   std::size_t max_iterations{1000000};
   bool update_dictionary{true};
   bool update_weights{true};
   bool require_monotonic_cost_decrease{true};

   std::tuple<bool, std::size_t, double> iterate(
      Factorization&) const;
};

template <class Factorization>
std::tuple<bool, std::size_t, double>
Factorization_delta_cost_iterator<Factorization>::iterate(
   Factorization& model) const
{
   using std::abs;

   double old_cost = model.cost();
   double new_cost = old_cost;
   double cost_delta = std::numeric_limits<double>::max();
   bool success = false;

   std::size_t iter = 0;
   while (iter < max_iterations) {
      old_cost = new_cost;

      if (update_dictionary) {
         model.update_dictionary();

         new_cost = model.cost();
         if (new_cost > old_cost && require_monotonic_cost_decrease) {
            throw std::runtime_error(
               "factorization cost increased after dictionary update");
         }
      }

      if (update_weights) {
         model.update_weights();

         new_cost = model.cost();
         if (new_cost > old_cost && require_monotonic_cost_decrease) {
            throw std::runtime_error(
               "factorization cost increased after weights update");
         }
      }

      cost_delta = new_cost - old_cost;

      ++iter;

      if (abs(cost_delta) < tolerance) {
         success = true;
         break;
      }
   }

   return std::make_tuple(success, iter, new_cost);
}

template <class Factorization>
struct VPFactorization_delta_cost_iterator {
   double tolerance{1e-8};
   std::size_t max_iterations{1000000};
   bool update_dictionary{true};
   bool update_weights{true};
   bool update_parameters{true};
   bool require_monotonic_cost_decrease{true};

   std::tuple<bool, std::size_t, double> iterate(
      Factorization&) const;
};

template <class Factorization>
std::tuple<bool, std::size_t, double>
VPFactorization_delta_cost_iterator<Factorization>::iterate(
   Factorization& model) const
{
   using std::abs;

   double old_cost = model.cost();
   double new_cost = old_cost;
   double cost_delta = std::numeric_limits<double>::max();
   bool success = false;

   std::size_t iter = 0;
   while (iter < max_iterations) {
      old_cost = new_cost;

      if (update_dictionary) {
         model.update_dictionary();

         new_cost = model.cost();
         if (new_cost > old_cost && require_monotonic_cost_decrease) {
            throw std::runtime_error(
               "factorization cost increased after dictionary update");
         }
      }

      if (update_weights) {
         model.update_weights();

         new_cost = model.cost();
         if (new_cost > old_cost && require_monotonic_cost_decrease) {
            throw std::runtime_error(
               "factorization cost increased after weights update");
         }
      }

      if (update_parameters) {
         model.update_parameters(model.get_data(), model.get_dictionary(),
                                 model.get_weights());

         new_cost = model.cost();
         if (new_cost > old_cost && require_monotonic_cost_decrease) {
            throw std::runtime_error(
               "factorization cost increased after parameters update");
         }
      }

      cost_delta = new_cost - old_cost;

      ++iter;

      if (abs(cost_delta) < tolerance) {
         success = true;
         break;
      }
   }

   return std::make_tuple(success, iter, new_cost);
}

} // namespace detail

template<class Factorization>
std::tuple<bool, int, double> iterate_factors_until_delta_cost_converged(
   Factorization& model, double tolerance, std::size_t max_iterations,
   bool update_dictionary, bool update_weights)
{
   detail::Factorization_delta_cost_iterator<Factorization> solver;
   solver.tolerance = tolerance;
   solver.max_iterations = max_iterations;
   solver.update_dictionary = update_dictionary;
   solver.update_weights = update_weights;

   return solver.iterate(model);
}

template <class Factorization>
std::tuple<bool, int, double> iterate_factors_until_delta_cost_converged(
   Factorization& model, double tolerance, std::size_t max_iterations,
   bool update_dictionary, bool update_weights, bool update_parameters)
{
   detail::VPFactorization_delta_cost_iterator<Factorization> solver;
   solver.tolerance = tolerance;
   solver.max_iterations = max_iterations;
   solver.update_dictionary = update_dictionary;
   solver.update_weights = update_weights;
   solver.update_parameters = update_parameters;

   return solver.iterate(model);
}

} // namespace reor

#endif
