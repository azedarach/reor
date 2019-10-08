#ifndef REOR_RANDOM_MATRIX_HPP_INCLUDED
#define REOR_RANDOM_MATRIX_HPP_INCLUDED

/**
 * @file random_matrix.hpp
 * @brief provide helper routines for generating stochastic matrices
 */

#include "backend_interface.hpp"

#include <random>
#include <vector>

namespace reor {

/**
 * @brief normalize matrix to be left stochastic
 * @tparam Matrix type of the matrix
 * @param A the matrix to be normalized
 */
template <class Matrix>
void make_left_stochastic_matrix(Matrix& A)
{
   using Scalar = typename backends::matrix_traits<Matrix>::element_type;

   const std::size_t n_rows = backends::rows(A);
   const std::size_t n_cols = backends::cols(A);

   std::vector<Scalar> col_sums(n_cols, 0);
   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         col_sums[j] += backends::get_matrix_element(i, j, A);
      }
   }

   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto value = backends::get_matrix_element(i, j, A);
         backends::set_matrix_element(i, j, value / col_sums[j], A);
      }
   }
}

/**
 * @brief normalize matrix to be right stochastic
 * @tparam Matrix type of the matrix
 * @param A the matrix to be normalized
 */
template <class Matrix>
void make_right_stochastic_matrix(Matrix& A)
{
   using Scalar = typename backends::matrix_traits<Matrix>::element_type;

   const std::size_t n_rows = backends::rows(A);
   const std::size_t n_cols = backends::cols(A);

   std::vector<Scalar> row_sums(n_rows, 0);
   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         row_sums[i] += backends::get_matrix_element(i, j, A);
      }
   }

   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto value = backends::get_matrix_element(i, j, A);
         backends::set_matrix_element(i, j, value / row_sums[i], A);
      }
   }
}

/**
 * @brief fill random left stochastic matrix
 * @tparam Matrix the type of the matrix
 * @tparam Generator the type of the pseudo-random number generator
 * @param A matrix to store the result
 * @param generator pseudo-random number generator
 */
template <class Matrix, class Generator>
void random_left_stochastic_matrix(Matrix& A, Generator& generator)
{
   using Real = typename backends::matrix_traits<Matrix>::real_element_type;

   std::uniform_real_distribution<Real> dist(0, 1);

   const std::size_t n_rows = backends::rows(A);
   const std::size_t n_cols = backends::cols(A);

   std::vector<Real> col_sums(n_cols, 0);
   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto aij = dist(generator);
         backends::set_matrix_element(i, j, aij, A);
         col_sums[j] += aij;
      }
   }

   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto value = backends::get_matrix_element(i, j, A);
         backends::set_matrix_element(i, j, value / col_sums[j], A);
      }
   }
}

/**
 * @brief fill random right stochastic matrix
 * @tparam Matrix the type of the matrix
 * @tparam Generator the type of the pseudo-random number generator
 * @param A matrix to store the result
 * @param generator pseudo-random number generator
 */
template <class Matrix, class Generator>
void random_right_stochastic_matrix(Matrix& A, Generator& generator)
{
   using Real = typename backends::matrix_traits<Matrix>::real_element_type;

   std::uniform_real_distribution<Real> dist(0, 1);

   const std::size_t n_rows = backends::rows(A);
   const std::size_t n_cols = backends::cols(A);

   std::vector<Real> row_sums(n_rows, 0);
   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto aij = dist(generator);
         backends::set_matrix_element(i, j, aij, A);
         row_sums[i] += aij;
      }
   }

   for (std::size_t j = 0; j < n_cols; ++j) {
      for (std::size_t i = 0; i < n_rows; ++i) {
         const auto value = backends::get_matrix_element(i, j, A);
         backends::set_matrix_element(i, j, value / row_sums[i], A);
      }
   }
}

} // namespace reor

#endif
