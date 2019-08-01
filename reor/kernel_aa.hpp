#ifndef REOR_KERNEL_AA_HPP_INCLUDED
#define REOR_KERNEL_AA_HPP_INCLUDED

/**
 * @file kernel_aa.hpp
 * @brief contains definition of kernelized archetypal analysis
 */

#include "backend_interface.hpp"
#include "numerics_helpers.hpp"

#include <deque>
#include <stdexcept>

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
   Matrix CtKC{};

   Matrix C_old{};
   Matrix grad_C{};
   Matrix incr_C{};

   Matrix diag_alpha_old{};
   Matrix grad_alpha{};
   Matrix incr_alpha{};

   Matrix S_old{};
   Matrix grad_S{};
   Matrix incr_S{};

   double beta{0.5};
   double sigma{0.01};
   double alpha_C{1};
   double alpha_S{1};
   double alpha_scale_factors{1};

   void update_dictionary_gradient();
   int dictionary_line_search();
   void update_weights_gradient();
   int weights_line_search();
   void update_scale_factors_gradient();
   int scale_factors_line_search();
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
   CtKC = Backend::create_matrix(n_components, n_components);

   C_old = Backend::copy_matrix(C_);
   grad_C = Backend::create_matrix(n_samples, n_components);
   incr_C = Backend::create_matrix(n_samples, n_components);

   diag_alpha_old = Backend::copy_matrix(diag_alpha);
   grad_alpha = Backend::create_diagonal_matrix(n_components, 0);
   incr_alpha = Backend::create_diagonal_matrix(n_components, 0);

   S_old = Backend::copy_matrix(S_);
   grad_S = Backend::create_matrix(n_components, n_samples);
   incr_S = Backend::create_matrix(n_components, n_samples);
}

template <class Backend>
void KernelAA<Backend>::set_line_search_beta(double b)
{
   if (b <= 0 || b >= 1) {
      throw std::runtime_error(
         "step-size reduction factor must be between 0 and 1");
   }
   beta = b;
}

template <class Backend>
void KernelAA<Backend>::set_line_search_sigma(double s)
{
   if (s < 0 || s > 1) {
      throw std::runtime_error(
         "function decrease factor must be between 0 and 1");
   }
   sigma = s;
}

template <class Backend>
double KernelAA<Backend>::cost()
{
   backends::gemm(1, S, S, 1, SSt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   backends::gemm(1, K, C, 1, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }

   backends::gemm(1, C, KC, 0, CtKC,
                  backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, CtKC, 0, CtKC);
   }

   return backends::trace_gemm_op(1, CtKC, SSt)
      + backends::trace_gemm_op(-2, KC, S) + trace_K;
}

template <class Backend>
void KernelAA<Backend>::update_weights_gradient()
{
   const std::size_t n_samples = backends::cols(S);
   const double normalization = n_samples / trace_K;

   backends::gemm(1, C, K, 0, grad_S,
                  backends::Op_flag::Transpose, backends::Op_flag::Transpose);
   if (delta > 0) {
      backends::gemm(1, diag_alpha, grad_S, 0, grad_S);
   }
   backends::gemm(normalization, CtKC, S, -normalization, grad_S);
}

template <class Backend>
int KernelAA<Backend>::weights_line_search()
{
   const double safety = std::numeric_limits<double>::min();

   const double current_cost = cost();
   S_old = Backend::copy_matrix(S);

   const double incr_norm = backends::sum_hadamard_op(1, incr_S, incr_S);

   int error = 0;
   bool finished_searching = false;
   while (!finished_searching) {
      backends::geam(1, S_old, -alpha_S, incr_S, S);
      backends::threshold_min(S, 0);
      backends::normalize_columns_by_lpnorm(S, 1, safety);

      const double next_cost = cost();
      if (next_cost <= current_cost - sigma * alpha_S * incr_norm) {
         alpha_S /= beta;
         finished_searching = true;
      } else {
         alpha_S *= beta;
      }

      if (is_zero(alpha_S)) {
         finished_searching = true;
      }
   }

   return error;
}

template <class Backend>
int KernelAA<Backend>::update_weights()
{
   const std::size_t n_components = backends::rows(S);
   const std::size_t n_samples = backends::cols(S);

   backends::gemm(1, S, S, 1, SSt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   // update gradient of cost function with respect to weights
   update_weights_gradient();

   // compute next increment
   Matrix projector = Backend::create_constant_matrix(n_components, 1, 1);
   Matrix sums = Backend::create_matrix(1, n_samples);
   backends::gemm(1, grad_S, S, 0, incr_S,
                  backends::Op_flag::Transpose);
   for (std::size_t i = 0; i < n_samples; ++i) {
      backends::set_matrix_element(
         0, i, backends::get_matrix_element(i, i, incr_S), sums);
   }
   backends::gemm(1, projector, sums, 0, incr_S);
   backends::geam(1, grad_S, -1, incr_S, incr_S);

   // update weights by line search
   const int error = weights_line_search();

   return error;
}

template <class Backend>
void KernelAA<Backend>::update_dictionary_gradient()
{
   const double normalization = 1.0 / trace_K;

   backends::gemm(1, K, C, 1, KC);
   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }

   backends::gemm(1, K, S, 0, grad_C,
                  backends::Op_flag::None, backends::Op_flag::Transpose);
   backends::gemm(normalization, KC, SSt, -normalization, grad_C);

   if (delta > 0) {
      backends::gemm(1, grad_C, diag_alpha, 0, grad_C);
   }
}

template <class Backend>
int KernelAA<Backend>::dictionary_line_search()
{
   const double safety = std::numeric_limits<double>::min();

   const double current_cost = cost();
   C_old = Backend::copy_matrix(C);

   const double incr_norm = backends::sum_hadamard_op(1, incr_C, incr_C);

   int error = 0;
   bool finished_searching = false;
   while (!finished_searching) {
      backends::geam(1, C_old, -alpha_C, incr_C, C);
      backends::threshold_min(C, 0);
      backends::normalize_columns_by_lpnorm(C, 1, safety);

      const double next_cost = cost();
      if (next_cost <= current_cost - sigma * alpha_C * incr_norm) {
         alpha_C /= beta;
         finished_searching = true;
      } else {
         alpha_C *= beta;
      }

      if (is_zero(alpha_C)) {
         finished_searching = true;
      }
   }

   return error;
}

template <class Backend>
void KernelAA<Backend>::update_scale_factors_gradient()
{
   const std::size_t n_samples = backends::rows(C);
   const std::size_t n_components = backends::cols(C);
   const double normalization = 1.0 / (trace_K * n_samples);

   backends::gemm(1, K, C, 1, KC);

   Matrix grad_mat = Backend::create_matrix(n_components, n_components);
   backends::gemm(1, S, KC, 0, grad_mat);

   if (delta > 0) {
      backends::gemm(1, KC, diag_alpha, 0, KC);
   }

   backends::gemm(1, C, KC, 0, CtKC, backends::Op_flag::Transpose);
   backends::gemm(normalization, CtKC, SSt, -normalization, grad_mat);

   for (std::size_t i = 0; i < n_components; ++i) {
      const auto g = backends::get_matrix_element(i, i, grad_mat);
      backends::set_matrix_element(i, i, g, grad_alpha);
   }
}

template <class Backend>
int KernelAA<Backend>::scale_factors_line_search()
{
   const double current_cost = cost();
   diag_alpha_old = Backend::copy_matrix(diag_alpha);

   const double incr_norm = backends::sum_hadamard_op(
      1, incr_alpha, incr_alpha);

   int error = 0;
   bool finished_searching = false;
   while (!finished_searching) {
      backends::geam(1, diag_alpha_old, -alpha_scale_factors,
                     incr_alpha, diag_alpha);

      backends::threshold_min(diag_alpha, 1 - delta);
      backends::threshold_max(diag_alpha, 1 + delta);

      const double next_cost = cost();
      if (next_cost <= current_cost - sigma * alpha_scale_factors * incr_norm) {
         alpha_scale_factors /= beta;
         finished_searching = true;
      } else {
         alpha_scale_factors *= beta;
      }

      if (is_zero(alpha_scale_factors)) {
         finished_searching = true;
      }
   }

   return error;
}

template <class Backend>
int KernelAA<Backend>::update_dictionary()
{
   const std::size_t n_samples = backends::rows(C);
   const std::size_t n_components = backends::cols(C);

   // enforce initial normalization
   if (delta > 0) {
      for (std::size_t i = 0; i < n_components; ++i) {
         double sum = 0;
         for (std::size_t t = 0; t < n_samples; ++t) {
            sum += backends::get_matrix_element(t, i, C);
         }
         backends::set_matrix_element(i, i, sum, diag_alpha);
      }

      for (std::size_t i = 0; i < n_components; ++i) {
         const double sum = backends::get_matrix_element(i, i, diag_alpha);
         for (std::size_t t = 0; t < n_samples; ++t) {
            const double cti = backends::get_matrix_element(t, i, C);
            backends::set_matrix_element(t, i, cti / sum, C);
         }
      }
   }

   backends::gemm(1, S, S, 1, SSt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   // update gradient of cost function with respect to dictionary
   update_dictionary_gradient();

   // compute next increment
   Matrix projector = Backend::create_constant_matrix(n_samples, 1, 1);
   Matrix sums = Backend::create_matrix(1, n_components);
   backends::gemm(1, grad_C, C, 0, incr_C,
                  backends::Op_flag::Transpose);
   for (std::size_t i = 0; i < n_components; ++i) {
      backends::set_matrix_element(
         0, i, backends::get_matrix_element(i, i, incr_C), sums);
   }
   backends::gemm(1, projector, sums, 0, incr_C);
   backends::geam(1, grad_C, -1, incr_C, incr_C);

   // update dictionary element by line search
   int error = dictionary_line_search();

   if (delta > 0) {
      // update gradient of cost function with respect to scale factors
      update_scale_factors_gradient();

      // compute next increment
      incr_alpha = Backend::copy_matrix(grad_alpha);

      const int scales_error = scale_factors_line_search();

      error = error > scales_error ? error : scales_error;
   }

   return error;
}

} // namespace reor

#endif
