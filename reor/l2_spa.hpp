#ifndef REOR_L2_SPA_HPP_INCLUDED
#define REOR_L2_SPA_HPP_INCLUDED

/**
 * @file l2_spa.hpp
 * @brief contains definition of classes providing SPA discretizations
 */

#include "backend_interface.hpp"
#include "l2_spa_regularizations.hpp"
#include "numerics_helpers.hpp"
#include "spg.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <tuple>

namespace reor {

template <
   class Backend,
   class RegularizationPolicy = L2_SPA_No_regularization
   >
class L2_SPA : public RegularizationPolicy {
public:
   using Matrix = typename Backend::Matrix;
   using Real = typename backends::matrix_traits<Matrix>::real_element_type;

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
   L2_SPA(const DataMatrix&, const StatesMatrix&, const AffiliationsMatrix&);
   ~L2_SPA() = default;
   L2_SPA(const L2_SPA&) = default;
   L2_SPA(L2_SPA&&) = default;
   L2_SPA& operator=(const L2_SPA&) = default;
   L2_SPA& operator=(L2_SPA&&) = default;

   void set_line_search_parameters(const SPG_line_search_parameters<Real>& p) {
      line_search_parameters = p;
   }

   double cost() const;
   int update_dictionary();
   int update_affiliations();

   std::tuple<bool, int, double> iterate_until_cost_converged(double, int);
   std::tuple<bool, int, double> iterate_affiliations_until_cost_converged(
      double, int);

   const Matrix& get_data() const { return X; }
   const Matrix& get_dictionary() const { return S; }
   const Matrix& get_affiliations() const { return Gamma; }

private:
   Matrix X{};
   Matrix S{};
   Matrix Gamma{};

   Matrix GGt{};
   Matrix StS{};

   Matrix S_old{};
   Matrix grad_S{};
   Matrix incr_S{};
   Matrix delta_grad_S{};

   Matrix Gamma_old{};
   Matrix grad_Gamma{};
   Matrix incr_Gamma{};
   Matrix delta_grad_Gamma{};

   SPG_line_search_parameters<Real> line_search_parameters{};
   double alpha_S{1.};
   double alpha_Gamma{1.};
   std::size_t mem{1};
   std::deque<double> f_S_mem{};
   std::deque<double> f_Gamma_mem{};

   double loss_function() const;
   void update_dictionary_gradient();
   std::tuple<int, double> dictionary_line_search();
   void update_affiliations_gradient();
   std::tuple<int, double> affiliations_line_search();
};

template <class Backend, class RegularizationPolicy>
template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
L2_SPA<Backend, RegularizationPolicy>::L2_SPA(
   const DataMatrix& X_, const StatesMatrix& S_,
   const AffiliationsMatrix& Gamma_)
{
   const auto n_features = backends::rows(X_);
   const auto n_samples = backends::cols(X_);
   const auto n_components = backends::cols(S_);

   if (backends::rows(S_) != n_features) {
      throw std::runtime_error(
         "number of rows in dictionary does not match number of features");
   }

   if (backends::cols(Gamma_) != n_samples) {
      throw std::runtime_error(
         "number of columns in affiliations does not match number of samples");
   }

   if (backends::rows(Gamma_) != n_components) {
      throw std::runtime_error(
         "number of rows in affiliations does not match number of components");
   }

   X = Backend::copy_matrix(X_);
   S = Backend::copy_matrix(S_);
   Gamma = Backend::copy_matrix(Gamma_);

   GGt = Backend::create_matrix(n_components, n_components);
   StS = Backend::create_matrix(n_components, n_components);

   S_old = Backend::copy_matrix(S_);
   grad_S = Backend::create_matrix(n_features, n_components);
   delta_grad_S = Backend::create_matrix(n_features, n_components);
   incr_S = Backend::create_matrix(n_features, n_components);

   Gamma_old = Backend::copy_matrix(Gamma_);
   grad_Gamma = Backend::create_matrix(n_components, n_samples);
   delta_grad_Gamma = Backend::create_matrix(n_components, n_samples);
   incr_Gamma = Backend::create_matrix(n_components, n_samples);

   f_S_mem = std::deque<double>(mem, 0);
   f_Gamma_mem = std::deque<double>(mem, 0);
}

template <class Backend, class RegularizationPolicy>
double L2_SPA<Backend, RegularizationPolicy>::loss_function() const
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   const double residual = backends::matrix_residual_fro_norm(
      X, S, Gamma);

   return residual * residual * normalization;
}

template <class Backend, class RegularizationPolicy>
double L2_SPA<Backend, RegularizationPolicy>::cost() const
{
   return loss_function() + RegularizationPolicy::penalty(X, S, Gamma);
}

template <class Backend, class RegularizationPolicy>
void L2_SPA<Backend, RegularizationPolicy>::update_dictionary_gradient()
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   RegularizationPolicy::dictionary_gradient(X, S, Gamma, grad_S);
   backends::gemm(-2 * normalization, X, Gamma, 1, grad_S,
                  backends::Op_flag::None, backends::Op_flag::Transpose);
   backends::gemm(2 * normalization, S, GGt, 1, grad_S);
}

template <class Backend, class RegularizationPolicy>
std::tuple<int, double>
L2_SPA<Backend, RegularizationPolicy>::dictionary_line_search()
{
   const double current_cost = cost();
   S_old = Backend::copy_matrix(S);

   f_S_mem.pop_front();
   f_S_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_S_mem[i] > f_max) {
         f_max = f_S_mem[i];
      }
   }

   const double delta = backends::sum_hadamard_op(1, incr_S, grad_S);
   double lambda = 1;

   backends::geam(1, S_old, 1, incr_S, S);
   double next_cost = cost();

   int error = 0;
   while (next_cost > f_max + line_search_parameters.gamma * lambda * delta) {

      lambda = line_search_parameters.get_next_step_length(
         lambda, delta, current_cost, next_cost);

      backends::geam(1, S_old, lambda, incr_S, S);
      next_cost = cost();

      if (is_zero(lambda)) {
         break;
      }
   }

   return std::make_tuple(error, lambda);
}

template <class Backend, class RegularizationPolicy>
int L2_SPA<Backend, RegularizationPolicy>::update_dictionary()
{
   using std::min;
   using std::max;

   backends::gemm(1, Gamma, Gamma, 0, GGt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   // update dictionary gradient
   update_dictionary_gradient();

   // compute next increment
   backends::geam(1, S, -alpha_S, grad_S, incr_S);
   backends::geam(1, incr_S, -1, S, incr_S);

   // line search for single step of gradient descent
   const auto line_search_result = dictionary_line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   // update alpha for next iteration
   delta_grad_S = Backend::copy_matrix(grad_S);

   update_dictionary_gradient();

   backends::geam(1, grad_S, -1, delta_grad_S, delta_grad_S);

   const double sksk = backends::sum_hadamard_op(
      lambda * lambda, incr_S, incr_S);
   const double beta = backends::sum_hadamard_op(
      lambda, incr_S, delta_grad_S);

   alpha_S = line_search_parameters.get_next_alpha(beta, sksk);

   return error;
}

template <class Backend, class RegularizationPolicy>
void L2_SPA<Backend, RegularizationPolicy>::update_affiliations_gradient()
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   RegularizationPolicy::affiliations_gradient(X, S, Gamma, grad_Gamma);

   backends::gemm(-2 * normalization, S, X, 1, grad_Gamma,
                  backends::Op_flag::Transpose, backends::Op_flag::None);
   backends::gemm(2 * normalization, StS, Gamma, 1, grad_Gamma);
}

template <class Backend, class RegularizationPolicy>
std::tuple<int, double>
L2_SPA<Backend, RegularizationPolicy>::affiliations_line_search()
{
   const double current_cost = cost();
   Gamma_old = Backend::copy_matrix(Gamma);

   f_Gamma_mem.pop_front();
   f_Gamma_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_Gamma_mem[i] > f_max) {
         f_max = f_Gamma_mem[i];
      }
   }

   const double delta = backends::sum_hadamard_op(1, incr_Gamma, grad_Gamma);
   double lambda = 1;

   backends::geam(1, Gamma_old, 1, incr_Gamma, Gamma);
   double next_cost = cost();

   int error = 0;
   while (next_cost > f_max + line_search_parameters.gamma * lambda * delta) {

      lambda = line_search_parameters.get_next_step_length(
         lambda, delta, current_cost, next_cost);

      backends::geam(1, Gamma_old, lambda, incr_Gamma, Gamma);
      next_cost = cost();

      if (is_zero(lambda)) {
         break;
      }
   }

   return std::make_tuple(error, lambda);
}

template <class Backend, class RegularizationPolicy>
int L2_SPA<Backend, RegularizationPolicy>::update_affiliations()
{
   using std::min;
   using std::max;

   backends::gemm(1, S, S, 0, StS,
                  backends::Op_flag::Transpose, backends::Op_flag::None);

   // update gradient of cost function
   update_affiliations_gradient();

   // compute next increment
   backends::geam(1, Gamma, -alpha_Gamma, grad_Gamma, incr_Gamma);
   backends::simplex_project_columns(incr_Gamma);
   backends::geam(1, incr_Gamma, -1, Gamma, incr_Gamma);

   // line search for single step of projected gradient descent
   const auto line_search_result = affiliations_line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   // update alpha for next iteration
   delta_grad_Gamma = Backend::copy_matrix(grad_Gamma);

   update_affiliations_gradient();

   backends::geam(1, grad_Gamma, -1, delta_grad_Gamma, delta_grad_Gamma);

   const double sksk = backends::sum_hadamard_op(
      lambda * lambda, incr_Gamma, incr_Gamma);
   const double beta = backends::sum_hadamard_op(
      lambda, incr_Gamma, delta_grad_Gamma);

   alpha_Gamma = line_search_parameters.get_next_alpha(beta, sksk);

   return error;
}

template <class Backend, class RegularizationPolicy>
std::tuple<bool, int, double>
L2_SPA<Backend, RegularizationPolicy>::iterate_until_cost_converged(
   double tolerance, int max_iterations)
{
   double old_cost = cost();
   double new_cost = old_cost;
   double cost_delta = std::numeric_limits<double>::max();
   bool success = false;

   int iter = 0;
   while (iter < max_iterations) {
      old_cost = new_cost;

      update_dictionary();

      const double tmp_cost = cost();
      if (tmp_cost > old_cost) {
         throw std::runtime_error(
            "factorization cost increased after dictionary update");
      }

      update_affiliations();

      new_cost = cost();
      cost_delta = new_cost - old_cost;

      if (cost_delta > 0) {
         throw std::runtime_error(
            "factorization cost increased after affiliations update");
      }

      ++iter;

      if (std::abs(cost_delta) < tolerance) {
         success = true;
         break;
      }
   }

   return std::make_tuple(success, iter, new_cost);
}

template <class Backend, class RegularizationPolicy>
std::tuple<bool, int, double>
L2_SPA<Backend, RegularizationPolicy>::iterate_affiliations_until_cost_converged(
   double tolerance, int max_iterations)
{
   double old_cost = cost();
   double new_cost = old_cost;
   double cost_delta = std::numeric_limits<double>::max();
   bool success = false;

   int iter = 0;
   while (iter < max_iterations) {
      old_cost = new_cost;

      update_affiliations();

      new_cost = cost();
      cost_delta = new_cost - old_cost;

      if (cost_delta > 0) {
         throw std::runtime_error(
            "factorization cost increased after affiliations update");
      }

      if (std::abs(cost_delta) < tolerance) {
         success = true;
         break;
      }

      ++iter;
   }

   return std::make_tuple(success, iter, new_cost);
}

} // namespace reor

#endif
