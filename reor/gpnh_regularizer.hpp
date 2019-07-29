#ifndef REOR_GPNH_REGULARIZER_HPP_INCLUDED
#define REOR_GPNH_REGULARIZER_HPP_INCLUDED

/**
 * @file gpnh_regularizer.hpp
 * @brief contains definition of GPNH regularization
 */

#include "backends.hpp"

namespace reor {

/**
 * @class GPNH_regularizer
 * @brief implements GPNH dictionary regularization
 */
template <class Backend>
class GPNH_regularizer {
public:

   void set_epsilon_states(double);
   double get_epsilon_states() const { return epsilon_S; }

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
   double penalty(const DataMatrix&, const StatesMatrix&,
                  const WeightsMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   void dictionary_gradient(const DataMatrix&, const StatesMatrix&,
                            const WeightsMatrix&, JacobianMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   void weights_gradient(const DataMatrix&, const StatesMatrix&,
                              const WeightsMatrix&, JacobianMatrix&) const;

protected:
   ~GPNH_regularizer() = default;

private:
   double epsilon_S{0};
};

template <class Backend>
void GPNH_regularizer<Backend>::set_epsilon_states(double eps)
{
   if (eps < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   epsilon_S = eps;
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
double GPNH_regularizer<Backend>::penalty(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const WeightsMatrix& /* Gamma */) const
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
template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void GPNH_regularizer<Backend>::dictionary_gradient(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const WeightsMatrix& /* Gamma */, JacobianMatrix& jac_S) const
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
template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void GPNH_regularizer<Backend>::weights_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& /* Gamma */, JacobianMatrix& jac_Gamma) const
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
