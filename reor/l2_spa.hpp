#ifndef REOR_L2_SPA_HPP_INCLUDED
#define REOR_L2_SPA_HPP_INCLUDED

/**
 * @file l2_spa.hpp
 * @brief contains definition of classes providing SPA discretizations
 */

#include "backend_interface.hpp"
#include "l2_spa_regularizations.hpp"
#include "numerics_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <tuple>

namespace reor {

template <
   class Backend,
   class RegularizationPolicy = L2_SPA_No_regularization
   >
class L2_SPA : public RegularizationPolicy {
public:
   using Matrix = typename Backend::Matrix;

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
   L2_SPA(const DataMatrix&, const StatesMatrix&, const AffiliationsMatrix&);
   ~L2_SPA() = default;
   L2_SPA(const L2_SPA&) = default;
   L2_SPA(L2_SPA&&) = default;
   L2_SPA& operator=(const L2_SPA&) = default;
   L2_SPA& operator=(L2_SPA&&) = default;

   double cost() const;
   int update_dictionary();
   int update_affiliations();

   const Matrix& get_data() const { return X; }
   const Matrix& get_dictionary() const { return S; }
   const Matrix& get_affiliations() const { return Gamma; }

private:
   Matrix X{};
   Matrix S{};
   Matrix Gamma{};

   Matrix GGt{};
   Matrix XGt{};
   Matrix StS{};
   Matrix Gamma_old{};
   Matrix grad_Gamma{};
   Matrix incr_Gamma{};
   Matrix delta_grad_Gamma{};

   double alpha{1.};
   double alpha_min{1e-3};
   double alpha_max{1};
   double sigma_1{0.1};
   double sigma_2{0.9};
   double gamma{1e-4};
   std::size_t mem{3};
   std::deque<double> f_mem{};

   double loss_function() const;
   void update_affiliations_gradient();
   std::tuple<int, double> line_search();
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
   XGt = Backend::create_matrix(n_features, n_components);
   StS = Backend::create_matrix(n_components, n_components);

   Gamma_old = Backend::copy_matrix(Gamma_);
   grad_Gamma = Backend::create_matrix(n_components, n_samples);
   delta_grad_Gamma = Backend::create_matrix(n_components, n_samples);
   incr_Gamma = Backend::create_matrix(n_components, n_samples);

   f_mem = std::deque<double>(mem, 0);
}

template <class Backend, class RegularizationPolicy>
double L2_SPA<Backend, RegularizationPolicy>::loss_function() const
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double normalization = 1.0 / (n_features * n_samples);

   const double residual = backends::matrix_residual_fro_norm(
      X, S, Gamma);

   return residual * normalization;
}

template <class Backend, class RegularizationPolicy>
double L2_SPA<Backend, RegularizationPolicy>::cost() const
{
   return loss_function() + RegularizationPolicy::penalty(X, S, Gamma);
}


template <class Backend, class RegularizationPolicy>
int L2_SPA<Backend, RegularizationPolicy>::update_dictionary()
{
   const auto n_features = backends::rows(X);
   const auto n_samples = backends::cols(X);
   const double inv_normalization = n_features * n_samples;

   backends::gemm(1, Gamma, Gamma, 0, GGt,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   RegularizationPolicy::dictionary_gradient(X, S, Gamma, S);
   backends::gemm(1, X, Gamma, -0.5 * inv_normalization, S,
                  backends::Op_flag::None, backends::Op_flag::Transpose);

   const auto error = backends::solve_square_qr_right(GGt, S);

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
std::tuple<int, double> L2_SPA<Backend, RegularizationPolicy>::line_search()
{
   const double current_cost = cost();
   Gamma_old = Backend::copy_matrix(Gamma);

   f_mem.pop_front();
   f_mem.push_back(current_cost);

   double f_max = -std::numeric_limits<double>::max();
   for (std::size_t i = 0; i < mem; ++i) {
      if (f_mem[i] > f_max) {
         f_max = f_mem[i];
      }
   }

   const double delta = backends::sum_gemm_op(1, incr_Gamma, grad_Gamma);
   double lambda = 1;

   backends::geam(1, Gamma_old, 1, incr_Gamma, Gamma);
   double next_cost = cost();

   int error = 0;
   while (next_cost > f_max + gamma * lambda * delta) {

      const double lambda_tmp = -0.5 * lambda * lambda * delta /
         (next_cost - current_cost - lambda * delta);

      if (lambda_tmp >= sigma_1 && lambda_tmp <= sigma_2 * lambda) {
         lambda = lambda_tmp;
      } else {
         lambda *= 0.5;
      }

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
   backends::geam(1, Gamma, -alpha, grad_Gamma, incr_Gamma);
   backends::simplex_project_columns(incr_Gamma);
   backends::geam(1, incr_Gamma, -1, Gamma, incr_Gamma);

   // line search for single step of projected gradient descent
   const auto line_search_result = line_search();
   int error = std::get<0>(line_search_result);
   const double lambda = std::get<1>(line_search_result);

   // update alpha for next iteration
   delta_grad_Gamma = Backend::copy_matrix(grad_Gamma);

   update_affiliations_gradient();

   backends::geam(1, grad_Gamma, -1, delta_grad_Gamma, delta_grad_Gamma);

   const double sksk = backends::sum_gemm_op(
      lambda * lambda, incr_Gamma, incr_Gamma);
   const double beta = backends::sum_gemm_op(
      lambda, incr_Gamma, delta_grad_Gamma);

   if (beta <= 0) {
      alpha = alpha_max;
   } else {
      alpha = min(alpha_max, max(alpha_min, sksk / beta));
   }

   return error;
}

} // namespace reor

#endif
