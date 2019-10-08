#ifndef REOR_CONVEX_CODING_HPP_INCLUDED
#define REOR_CONVEX_CODING_HPP_INCLUDED

#include "backend_interface.hpp"
#include "numerics_helpers.hpp"
#include "spg.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <tuple>

namespace reor {

template <class Backend>
class GPNH_L2_SPA {
public:
   using Matrix = typename Backend::Matrix;
   using Real = typename backends::matrix_traits<Matrix>::real_element_type;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
   GPNH_L2_SPA(const DataMatrix&, const StatesMatrix&, const WeightsMatrix&);
   ~GPNH_L2_SPA() = default;
   GPNH_L2_SPA(const GPNH_L2_SPA&) = default;
   GPNH_L2_SPA(GPNH_L2_SPA&&) = default;
   GPNH_L2_SPA& operator=(const GPNH_L2_SPA&) = default;
   GPNH_L2_SPA& operator=(GPNH_L2_SPA&&) = default;

   double get_epsilon_states() const { return epsilon_S; }
   void set_epsilon_states(double eps);
   void set_line_search_parameters(const SPG_line_search_parameters<Real>& p) {
      line_search_parameters = p;
   }

   double cost() const;

   int update_dictionary();
   int update_weights();

   const Matrix& get_data() const { return X; }
   const Matrix& get_dictionary() const { return S; }
   const Matrix& get_weights() const { return Gamma; }

private:
   Matrix X{};
   Matrix S{};
   Matrix Gamma{};
   double epsilon_S{0};

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
   double penalty() const;
   void dictionary_penalty_gradient();
   void update_dictionary_gradient();
   std::tuple<int, double> dictionary_line_search();
   void weights_penalty_gradient();
   void update_weights_gradient();
   std::tuple<int, double> weights_line_search();
};

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
GPNH_L2_SPA<Backend>::GPNH_L2_SPA(
   const DataMatrix& X_, const StatesMatrix& S_,
   const WeightsMatrix& Gamma_)
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
         "number of columns in weights does not match number of components");
   }

   if (backends::rows(Gamma_) != n_components) {
      throw std::runtime_error(
         "number of rows in weights does not match number of components");
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

template <class Backend>
void GPNH_L2_SPA<Backend>::set_epsilon_states(double eps)
{
   if (eps < 0) {
      throw std::runtime_error(
         "dictionary regularization parameter must be non-negative");
   }

   epsilon_S = eps;
}

template <class Backend>
double GPNH_L2_SPA<Backend>::loss_function() const
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   const double residual = backends::matrix_residual_fro_norm(
      X, S, Gamma);

   return residual * residual * normalization;
}

template <class Backend>
double GPNH_L2_SPA<Backend>::penalty() const
{
   double value = 0;

   const std::size_t n_components = backends::cols(S);
   const std::size_t n_features = backends::rows(S);

   if (n_components == 1) {
      return value;
   }

   const double prefactor = 2.0 / (n_components * n_features *
                                   (n_components - 1));
   value += prefactor * n_components * backends::trace_gemm_op(
      1, S, S, backends::Op_flag::Transpose)
      - prefactor * backends::sum_gemm_op(
         1, S, S, backends::Op_flag::Transpose, backends::Op_flag::None);

   return epsilon_S * value;
}

template <class Backend>
double GPNH_L2_SPA<Backend>::cost() const
{
   return loss_function() + penalty();
}

template <class Backend>
void GPNH_L2_SPA<Backend>::dictionary_penalty_gradient()
{
   const std::size_t n_components = backends::cols(S);
   const std::size_t n_features = backends::rows(S);

   if (n_components == 1) {
      for (std::size_t i = 0; i < n_features; ++i) {
         for (std::size_t j = 0; j < n_components; ++j) {
            backends::set_matrix_element(i, j, 0, grad_S);
         }
      }
   } else {
      const double prefactor = 4.0 * epsilon_S /
         (n_features * n_components * (n_components - 1));

      auto D = Backend::create_diagonal_matrix(
         n_components, n_components);
      backends::add_constant(-1, D);

      backends::gemm(prefactor, S, D, 0, grad_S);
   }
}

template <class Backend>
void GPNH_L2_SPA<Backend>::update_dictionary_gradient()
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   dictionary_penalty_gradient();

   backends::gemm(-2 * normalization, X, Gamma, 1, grad_S,
                  backends::Op_flag::None, backends::Op_flag::Transpose);
   backends::gemm(2 * normalization, S, GGt, 1, grad_S);
}

template <class Backend>
std::tuple<int, double> GPNH_L2_SPA<Backend>::dictionary_line_search()
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

template <class Backend>
int GPNH_L2_SPA<Backend>::update_dictionary()
{
   backends::gemm(1, Gamma, Gamma, 0, GGt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

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

template <class Backend>
void GPNH_L2_SPA<Backend>::weights_penalty_gradient()
{
   const std::size_t n_components = backends::rows(grad_Gamma);
   const std::size_t n_samples = backends::cols(grad_Gamma);

   for (std::size_t i = 0; i < n_components; ++i) {
      for (std::size_t j = 0; j < n_samples; ++j) {
         backends::set_matrix_element(i, j, 0, grad_Gamma);
      }
   }
}

template <class Backend>
void GPNH_L2_SPA<Backend>::update_weights_gradient()
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   weights_penalty_gradient();

   backends::gemm(-2 * normalization, S, X, 1, grad_Gamma,
                  backends::Op_flag::Transpose, backends::Op_flag::None);
   backends::gemm(2 * normalization, StS, Gamma, 1, grad_Gamma);
}

template <class Backend>
std::tuple<int, double> GPNH_L2_SPA<Backend>::weights_line_search()
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

template <class Backend>
int GPNH_L2_SPA<Backend>::update_weights()
{
   backends::gemm(1, S, S, 0, StS,
                  backends::Op_flag::Transpose, backends::Op_flag::None);

   update_weights_gradient();

   // compute next increment
   backends::geam(1, Gamma, -alpha_Gamma, grad_Gamma, incr_Gamma);
   backends::simplex_project_columns(incr_Gamma);
   backends::geam(1, incr_Gamma, -1, Gamma, incr_Gamma);

   // line search for single step of projected gradient descent
   const auto line_search_result = weights_line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   // update alpha for next iteration
   delta_grad_Gamma = Backend::copy_matrix(grad_Gamma);

   update_weights_gradient();

   backends::geam(1, grad_Gamma, -1, delta_grad_Gamma, delta_grad_Gamma);

   const double sksk = backends::sum_hadamard_op(
      lambda * lambda, incr_Gamma, incr_Gamma);
   const double beta = backends::sum_hadamard_op(
      lambda, incr_Gamma, delta_grad_Gamma);

   alpha_Gamma = line_search_parameters.get_next_alpha(beta, sksk);

   return error;
}

} // namespace reor

#endif
