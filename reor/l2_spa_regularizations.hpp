#ifndef REOR_L2_SPA_REGULARIZATIONS_HPP_INCLUDED
#define REOR_L2_SPA_REGULARIZATIONS_HPP_INCLUDED

/**
 * @file l2_spa_regularizations.hpp
 * @brief contains definitions of regularizations for l2-SPA
 */

#include "backends.hpp"

namespace reor {

/**
 * @class L2_SPA_No_regularization
 */
struct L2_SPA_No_regularization {
   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
   static double penalty(const DataMatrix&, const StatesMatrix&,
                         const AffiliationsMatrix&);

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
             class JacobianMatrix>
   static void dictionary_gradient(const DataMatrix&, const StatesMatrix&,
                                   const AffiliationsMatrix&,
                                   JacobianMatrix&);

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
             class JacobianMatrix>
   static void affiliations_gradient(const DataMatrix&, const StatesMatrix&,
                                     const AffiliationsMatrix&,
                                     JacobianMatrix&);
};

template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
double L2_SPA_No_regularization::penalty(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const AffiliationsMatrix& /* Gamma */)
{
   return 0;
}

template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
          class JacobianMatrix>
void L2_SPA_No_regularization::dictionary_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const AffiliationsMatrix& /* Gamma */,
   JacobianMatrix& jac_S)
{
   const std::size_t n_features = backends::rows(jac_S);
   const std::size_t n_components = backends::cols(jac_S);

   for (std::size_t i = 0; i < n_features; ++i) {
      for (std::size_t j = 0; j < n_components; ++j) {
         backends::set_matrix_element(i, j, 0, jac_S);
      }
   }
}

template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
          class JacobianMatrix>
void L2_SPA_No_regularization::affiliations_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const AffiliationsMatrix& /* Gamma */,
   JacobianMatrix& jac_Gamma)
{
   const std::size_t n_components = backends::rows(jac_Gamma);
   const std::size_t n_samples = backends::cols(jac_Gamma);

   for (std::size_t i = 0; i < n_components; ++i) {
      for (std::size_t j = 0; j < n_samples; ++j) {
         backends::set_matrix_element(i, j, 0, jac_Gamma);
      }
   }
}

template <class Backend>
class L2_SPA_GPNH_regularization {
public:

   void set_epsilon_states(double);
   double get_epsilon_states() const { return epsilon_S; }

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
   double penalty(const DataMatrix&, const StatesMatrix&,
                  const AffiliationsMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
             class JacobianMatrix>
   void dictionary_gradient(const DataMatrix&, const StatesMatrix&,
                            const AffiliationsMatrix&, JacobianMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
             class JacobianMatrix>
   void affiliations_gradient(const DataMatrix&, const StatesMatrix&,
                              const AffiliationsMatrix&, JacobianMatrix&) const;

protected:
   ~L2_SPA_GPNH_regularization() = default;

private:
   double epsilon_S{0};
};

template <class Backend>
void L2_SPA_GPNH_regularization<Backend>::set_epsilon_states(double eps)
{
   if (eps < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   epsilon_S = eps;
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix>
double L2_SPA_GPNH_regularization<Backend>::penalty(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const AffiliationsMatrix& /* Gamma */) const
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
template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
          class JacobianMatrix>
void L2_SPA_GPNH_regularization<Backend>::dictionary_gradient(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const AffiliationsMatrix& /* Gamma */, JacobianMatrix& jac_S) const
{
   const std::size_t n_components = backends::cols(S);
   const std::size_t n_features = backends::rows(S);

   if (n_components == 1) {
      for (std::size_t i = 0; i < n_features; ++i) {
         for (std::size_t j = 0; j < n_components; ++j) {
            backends::set_matrix_element(i, j, 0, jac_S);
         }
      }
   } else {
      const double prefactor = 4.0 * epsilon_S /
         (n_features * n_components * (n_components - 1));

      auto D = Backend::create_diagonal_matrix(
         n_components, n_components);
      backends::add_constant(-1, D);

      backends::gemm(prefactor, S, D, 0, jac_S);
   }
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class AffiliationsMatrix,
          class JacobianMatrix>
void L2_SPA_GPNH_regularization<Backend>::affiliations_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const AffiliationsMatrix& /* Gamma */, JacobianMatrix& jac_Gamma) const
{
   const std::size_t n_components = backends::rows(jac_Gamma);
   const std::size_t n_samples = backends::cols(jac_Gamma);

   for (std::size_t i = 0; i < n_components; ++i) {
      for (std::size_t j = 0; j < n_samples; ++j) {
         backends::set_matrix_element(i, j, 0, jac_Gamma);
      }
   }
}

} // namespace reor

#endif
