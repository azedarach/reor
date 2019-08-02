#ifndef REOR_KERNEL_AA_HPP_INCLUDED
#define REOR_KERNEL_AA_HPP_INCLUDED

/**
 * @file kernel_aa.hpp
 * @brief contains definition of kernelized archetypal analysis
 */

#include "backend_interface.hpp"
#include "numerics_helpers.hpp"
#include "spg.hpp"

#include <deque>
#include <stdexcept>
#include <tuple>

namespace reor {

template <class Backend>
class KernelAA {
public:
   using Matrix = typename Backend::Matrix;
   using Real = typename backends::matrix_traits<Matrix>::real_element_type;

   template <class KernelMatrix, class StatesMatrix, class WeightsMatrix>
   KernelAA(const KernelMatrix&, const StatesMatrix&, const WeightsMatrix&,
            double delta_ = 0);
   ~KernelAA() = default;
   KernelAA(const KernelAA&) = default;
   KernelAA(KernelAA&&) = default;
   KernelAA& operator=(const KernelAA&) = default;
   KernelAA& operator=(KernelAA&) = default;

   void set_line_search_beta(double);
   void set_line_search_sigma(double);
   double get_delta() const { return delta; }

   double cost();

   int update_dictionary();
   int update_weights();

   const Matrix& get_kernel() const { return K; }
   const Matrix& get_dictionary() const { return C; }
   const Matrix& get_weights() const { return S; }
   const Matrix& get_scale_factors() const { return diag_alpha; }

private:
   double delta{0};
   double trace_K{0};
   Matrix K{};
   Matrix C{};
   Matrix S{};
   Matrix diag_alpha{};

   Matrix SSt{};
   Matrix KC{};
   Matrix SKC{};
   Matrix CtKC{};

   Matrix C_old{};
   Matrix grad_C{};
   Matrix incr_C{};
   Matrix delta_grad_C{};

   Matrix diag_alpha_old{};
   Matrix grad_alpha{};
   Matrix incr_alpha{};
   Matrix delta_grad_alpha{};

   Matrix S_old{};
   Matrix grad_S{};
   Matrix incr_S{};
   Matrix delta_grad_S{};

   SPG_line_search_parameters<Real> line_search_parameters{};
   double alpha_C{1};
   double alpha_S{1};
   double alpha_alpha{1};
   std::size_t mem{1};
   std::deque<double> f_C_mem{};
   std::deque<double> f_S_mem{};
   std::deque<double> f_alpha_mem{};

   void update_dictionary_gradient();
   std::tuple<int, double> dictionary_line_search();
   void update_weights_gradient();
   std::tuple<int, double> weights_line_search();
   void update_scale_factors_gradient();
   std::tuple<int, double> scale_factors_line_search();
};

template <class Backend>
template <class KernelMatrix, class StatesMatrix, class WeightsMatrix>
KernelAA<Backend>::KernelAA(
   const KernelMatrix& K_, const StatesMatrix& C_, const WeightsMatrix& S_,
   double delta_)
{
   const auto n_samples = backends::rows(K_);
   const auto n_components = backends::cols(C_);

   if (backends::rows(K_) != backends::cols(K_)) {
      throw std::runtime_error(
         "input kernel must be square");
   }

   if (backends::rows(S_) != n_components) {
      throw std::runtime_error(
         "number of rows in weights does not match number of archetypes");
   }

   if (delta_ < 0) {
      throw std::runtime_error(
         "relaxation parameter delta must be non-negative");
   }

   K = Backend::copy_matrix(K_);
   C = Backend::copy_matrix(C_);
   S = Backend::copy_matrix(S_);
   diag_alpha = Backend::create_diagonal_matrix(n_components, 1);

   delta = delta_;
   trace_K = backends::trace(K_);

   SSt = Backend::create_matrix(n_components, n_components);
   KC = Backend::create_matrix(n_samples, n_components);
   SKC = Backend::create_matrix(n_components, n_components);
   CtKC = Backend::create_matrix(n_components, n_components);

   C_old = Backend::copy_matrix(C_);
   grad_C = Backend::create_matrix(n_samples, n_components);
   delta_grad_C = Backend::create_matrix(n_components, n_samples);
   incr_C = Backend::create_matrix(n_samples, n_components);

   diag_alpha_old = Backend::copy_matrix(diag_alpha);
   grad_alpha = Backend::create_diagonal_matrix(n_components, 0);
   delta_grad_alpha = Backend::create_matrix(n_components, n_components);
   incr_alpha = Backend::create_diagonal_matrix(n_components, 0);

   S_old = Backend::copy_matrix(S_);
   grad_S = Backend::create_matrix(n_components, n_samples);
   delta_grad_S = Backend::create_matrix(n_components, n_samples);
   incr_S = Backend::create_matrix(n_components, n_samples);

   f_S_mem = std::deque<double>(mem, 0);
   f_C_mem = std::deque<double>(mem, 0);
   f_alpha_mem = std::deque<double>(mem, 0);
}

template <class Backend>
double KernelAA<Backend>::cost()
{
   backends::gemm(1, S, S, 0, SSt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   backends::gemm(1, K, C, 0, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }

   backends::gemm(1, C, KC, 0, CtKC,
                  backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, CtKC, 0, CtKC);
   }

   return backends::trace_gemm_op(1, CtKC, SSt)
      - 2 * backends::trace_gemm_op(1, KC, S) + trace_K;
}

template <class Backend>
void KernelAA<Backend>::update_weights_gradient()
{
   backends::gemm(1, C, K, 0, grad_S,
                  backends::Op_flag::Transpose, backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, grad_S, 0, grad_S);
   }
   backends::gemm(2, CtKC, S, -2, grad_S);
}

template <class Backend>
std::tuple<int, double> KernelAA<Backend>::weights_line_search()
{
   const double current_cost = trace_K
      - 2 * backends::trace_gemm_op(1, KC, S)
      + backends::trace_gemm_op(1, CtKC, SSt);
   S_old = Backend::copy_matrix(S);

   f_S_mem.pop_front();
   f_S_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_S_mem[i] > f_max) {
         f_max = f_S_mem[i];
      }
   }

   const double incr_norm = backends::sum_hadamard_op(1, incr_S, incr_S);
   double lambda = 1;

   backends::geam(1, S_old, 1, incr_S, S);

   backends::gemm(1, S, S, 0, SSt, backends::Op_flag::None,
                  backends::Op_flag::Transpose);

   double next_cost = trace_K - 2 * backends::trace_gemm_op(1, KC, S)
      + backends::trace_gemm_op(1, CtKC, SSt);

   int error = 0;
   const double factor = line_search_parameters.gamma * incr_norm;
   while (next_cost > f_max + lambda * factor) {
      lambda = line_search_parameters.get_next_step_length(
         lambda, incr_norm, current_cost, next_cost);

      backends::geam(1, S_old, lambda, incr_S, S);

      backends::gemm(1, S, S, 0, SSt, backends::Op_flag::None,
                     backends::Op_flag::Transpose);

      next_cost = trace_K - 2 * backends::trace_gemm_op(1, KC, S)
         + backends::trace_gemm_op(1, CtKC, SSt);

      if (is_zero(lambda)) {
         break;
      }
   }

   return std::make_tuple(error, lambda);
}

template <class Backend>
int KernelAA<Backend>::update_weights()
{
   backends::gemm(1, S, S, 0, SSt, backends::Op_flag::None,
                  backends::Op_flag::Transpose);

   backends::gemm(1, K, C, 0, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }

   backends::gemm(1, KC, C, 0, CtKC,
                  backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, CtKC, diag_alpha, 0, CtKC);
   }

   // update gradient of cost function with respect to weights
   update_weights_gradient();

   // compute next increment
   backends::geam(1, S, -alpha_S, grad_S, incr_S);
   backends::simplex_project_columns(incr_S);
   backends::geam(1, incr_S, -1, S, incr_S);

   // update weights by line search
   const auto line_search_result = weights_line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   delta_grad_S = Backend::copy_matrix(grad_S);

   update_weights_gradient();

   backends::geam(1, grad_S, -1, delta_grad_S, delta_grad_S);

   const double sksk = backends::sum_hadamard_op(
      lambda * lambda, incr_S, incr_S);
   const double beta = backends::sum_hadamard_op(
      lambda, incr_S, delta_grad_S);

   alpha_S = line_search_parameters.get_next_alpha(beta, sksk);

   return error;
}

template <class Backend>
void KernelAA<Backend>::update_dictionary_gradient()
{
   backends::gemm(1, K, S, 0, grad_C,
                  backends::Op_flag::Transpose,
                  backends::Op_flag::Transpose);

   backends::gemm(2, KC, SSt, -2, grad_C);
   if (delta > 0) {
      backends::gemm(1, grad_C, diag_alpha, 0, grad_C);
   }
}

template <class Backend>
std::tuple<int, double> KernelAA<Backend>::dictionary_line_search()
{
   const double current_cost = trace_K
      - 2 * backends::trace_gemm_op(1, KC, S)
      + backends::trace_gemm_op(1, SSt, CtKC);

   C_old = Backend::copy_matrix(C);

   f_C_mem.pop_front();
   f_C_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_C_mem[i] > f_max) {
         f_max = f_C_mem[i];
      }
   }

   const double incr_norm = backends::sum_hadamard_op(1, incr_C, incr_C);
   double lambda = 1;

   backends::geam(1, C_old, 1, incr_C, C);

   backends::gemm(1, K, C, 0, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }
   backends::gemm(1, C, KC, 0, CtKC,
                  backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, CtKC, 0, CtKC);
   }

   double next_cost = trace_K
      - 2 * backends::trace_gemm_op(1, KC, S)
      + backends::trace_gemm_op(1, SSt, CtKC);

   int error = 0;
   const double factor = line_search_parameters.gamma * incr_norm;
   while (next_cost > f_max + factor * lambda) {
      lambda = line_search_parameters.get_next_step_length(
         lambda, incr_norm, current_cost, next_cost);

      backends::geam(1, C_old, lambda, incr_C, C);

      backends::gemm(1, K, C, 0, KC);
      if (delta > 0) {
         backends::gemm(1, KC, diag_alpha, 0, KC);
      }
      backends::gemm(1, C, KC, 0, CtKC,
                     backends::Op_flag::Transpose);
      if (delta > 0) {
         backends::gemm(1, diag_alpha, CtKC, 0, CtKC);
      }

      next_cost = trace_K
         - 2 * backends::trace_gemm_op(1, KC, S)
         + backends::trace_gemm_op(1, SSt, CtKC);

      if (is_zero(lambda)) {
         break;
      }
   }

   return std::make_tuple(error, lambda);
}

template <class Backend>
void KernelAA<Backend>::update_scale_factors_gradient()
{
   const auto n_components = backends::rows(diag_alpha);
   Matrix grad_alpha_tmp = Backend::create_matrix(n_components, n_components);

   backends::gemm(1, diag_alpha, CtKC, 0, grad_alpha_tmp,
                  backends::Op_flag::None, backends::Op_flag::Transpose);
   backends::gemm(1, SSt, grad_alpha_tmp, 0, grad_alpha_tmp);

   backends::geam(-2, SKC, 2, grad_alpha_tmp, grad_alpha_tmp,
                  backends::Op_flag::Transpose);

   for (std::size_t i = 0; i < n_components; ++i) {
      const auto g = backends::get_matrix_element(i, i, grad_alpha_tmp);
      backends::set_matrix_element(i, i, g, grad_alpha);
   }
}

template <class Backend>
std::tuple<int, double> KernelAA<Backend>::scale_factors_line_search()
{
   Matrix CtKC_tmp = Backend::copy_matrix(CtKC);
   backends::gemm(1, diag_alpha, CtKC, 0, CtKC_tmp);
   backends::gemm(1, CtKC_tmp, diag_alpha, 0, CtKC_tmp);

   const double current_cost = trace_K
      - 2 * backends::trace_gemm_op(1, SKC, diag_alpha)
      + backends::trace_gemm_op(1, SSt, CtKC_tmp);

   diag_alpha_old = Backend::copy_matrix(diag_alpha);

   f_alpha_mem.pop_front();
   f_alpha_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_alpha_mem[i] > f_max) {
         f_max = f_alpha_mem[i];
      }
   }

   const double incr_norm = backends::sum_hadamard_op(
      1, incr_alpha, incr_alpha);
   double lambda = 1;

   backends::geam(1, diag_alpha_old, 1, incr_alpha, diag_alpha);

   backends::gemm(1, diag_alpha, CtKC, 0, CtKC_tmp);
   backends::gemm(1, CtKC_tmp, diag_alpha, 0, CtKC_tmp);

   double next_cost = trace_K
      - 2 * backends::trace_gemm_op(1, SKC, diag_alpha)
      + backends::trace_gemm_op(1, SSt, CtKC_tmp);

   int error = 0;
   const double factor = line_search_parameters.gamma * incr_norm;
   while (next_cost > f_max + factor * lambda) {
      lambda = line_search_parameters.get_next_step_length(
         lambda, incr_norm, current_cost, next_cost);

      backends::geam(1, diag_alpha_old, lambda, incr_alpha, diag_alpha);

      backends::gemm(1, diag_alpha, CtKC, 0, CtKC_tmp);
      backends::gemm(1, CtKC_tmp, diag_alpha, 0, CtKC_tmp);

      next_cost = trace_K
         - 2 * backends::trace_gemm_op(1, SKC, diag_alpha)
         + backends::trace_gemm_op(1, SSt, CtKC_tmp);

      if (is_zero(lambda)) {
         break;
      }
   }

   return std::make_tuple(error, lambda);
}

template <class Backend>
int KernelAA<Backend>::update_dictionary()
{
   backends::gemm(1, S, S, 0, SSt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   backends::gemm(1, K, C, 0, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }
   backends::gemm(1, C, KC, 0, CtKC,
                  backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, CtKC, 0, CtKC);
   }

   // update gradient of cost function with respect to dictionary
   update_dictionary_gradient();

   // compute next increment
   backends::geam(1, C, -alpha_C, grad_C, incr_C);
   backends::simplex_project_columns(incr_C);
   backends::geam(1, incr_C, -1, C, incr_C);

   // update dictionary element by line search
   const auto line_search_result = dictionary_line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   // update alpha for next iteration
   delta_grad_C = Backend::copy_matrix(grad_C);

   update_dictionary_gradient();

   backends::geam(1, grad_C, -1, delta_grad_C, delta_grad_C);

   const double sksk = backends::sum_hadamard_op(
      lambda * lambda, incr_C, incr_C);
   const double beta = backends::sum_hadamard_op(
      lambda, incr_C, delta_grad_C);

   alpha_C = line_search_parameters.get_next_alpha(beta, sksk);

   if (delta > 0) {
      backends::gemm(1, K, C, 0, KC);
      backends::gemm(1, S, KC, 0, SKC);
      backends::gemm(1, C, KC, 0, CtKC,
                     backends::Op_flag::Transpose);

      // update gradient of cost function with respect to scale factors
      update_scale_factors_gradient();

      backends::geam(1, diag_alpha, -alpha_alpha, grad_alpha, incr_alpha);
      backends::threshold_min(incr_alpha, 1 - delta);
      backends::threshold_max(incr_alpha, 1 + delta);
      backends::geam(1, incr_alpha, -1, diag_alpha, incr_alpha);

      const auto line_search_result = scale_factors_line_search();
      const int scales_error = std::get<0>(line_search_result);
      const double lambda = std::get<1>(line_search_result);

      delta_grad_alpha = Backend::copy_matrix(grad_alpha);

      update_scale_factors_gradient();

      backends::geam(1, grad_alpha, -1, delta_grad_alpha, delta_grad_alpha);

      const double sksk = backends::sum_hadamard_op(
         lambda * lambda, incr_alpha, incr_alpha);
      const double beta = backends::sum_hadamard_op(
         lambda, incr_alpha, delta_grad_alpha);

      alpha_alpha = line_search_parameters.get_next_alpha(beta, sksk);

      error = error > scales_error ? error : scales_error;
   }

   return error;
}

} // namespace reor

#endif
