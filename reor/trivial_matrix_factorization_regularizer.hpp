#ifndef REOR_TRIVIAL_MATRIX_FACTORIZATION_REGULARIZER_HPP_INCLUDED
#define REOR_TRIVIAL_MATRIX_FACTORIZATION_REGULARIZER_HPP_INCLUDED

/**
 * @file trivial_matrix_factorization_regularizer.hpp
 * @brief contains definition of trivial matrix factorization regularizer
 */

#include "backend_interface.hpp"

namespace reor {

/**
 * @class Trivial_matrix_factorization_regularizer
 * @brief provides trivial (i.e., no) regularization for matrix factors
 */
struct Trivial_matrix_factorization_regularizer {
   template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
   static double penalty(const DataMatrix&, const StatesMatrix&,
                         const WeightsMatrix&);

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   static void dictionary_gradient(const DataMatrix&, const StatesMatrix&,
                                   const WeightsMatrix&,
                                   JacobianMatrix&);

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   static void weights_gradient(const DataMatrix&, const StatesMatrix&,
                                const WeightsMatrix&,
                                JacobianMatrix&);
};

template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
double Trivial_matrix_factorization_regularizer::penalty(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& /* Gamma */)
{
   return 0;
}

template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void Trivial_matrix_factorization_regularizer::dictionary_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& /* Gamma */,
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

template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void Trivial_matrix_factorization_regularizer::weights_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& /* Gamma */,
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

} // namespace reor

#endif
