#ifndef REOR_COMPARISON_HELPERS_HPP_INCLUDED
#define REOR_COMPARISON_HELPERS_HPP_INCLUDED

#include "reor/backend_interface.hpp"

#include <limits>
#include <type_traits>

namespace detail {

template <class LhsMatrix, class RhsMatrix, class Enable = void>
struct common_real_type {};

template <class LhsMatrix, class RhsMatrix, class Scalar, class Enable = void>
struct is_equal_impl {};

template <class Matrix, class Real, class Enable = void>
struct is_right_stochastic_matrix_impl {};

template <class Matrix, class Real, class Enable = void>
struct is_left_stochastic_matrix_impl {};

} // namespace detail

#ifdef HAVE_EIGEN

#include <Eigen/Core>

#include "reor/backends/eigen_type_traits.hpp"

namespace detail {

template <class LhsMatrix, class RhsMatrix>
struct common_real_type<
   LhsMatrix, RhsMatrix,
   typename std::enable_if<reor::backends::detail::is_eigen_matrix<
                              LhsMatrix>::value &&
                           reor::backends::detail::is_eigen_matrix<
                              RhsMatrix>::value>::type> {
   using type = typename std::common_type<
      typename LhsMatrix::RealScalar, typename RhsMatrix::RealScalar>::type;
};

template <class LhsMatrix, class RhsMatrix, class Real>
struct is_equal_impl<
   LhsMatrix, RhsMatrix, Real,
   typename std::enable_if<reor::backends::detail::is_eigen_matrix<
                              LhsMatrix>::value &&
                           reor::backends::detail::is_eigen_matrix<
                              RhsMatrix>::value>::type> {

   static bool check(const LhsMatrix& a, const RhsMatrix& b, Real tol)
      {
         return (b - a).cwiseAbs().maxCoeff() < tol;
      }
};

template <class Matrix, class Real>
struct is_right_stochastic_matrix_impl<
   Matrix, Real,
   typename std::enable_if<
      reor::backends::detail::is_eigen_matrix<Matrix>::value
      >::type> {

   static bool check(const Matrix& m, Real tol)
      {
         const bool all_non_negative = (m.array() >= 0).all();

         const Eigen::VectorXd row_sums = m.rowwise().sum();
         const Eigen::VectorXd target(Eigen::VectorXd::Ones(row_sums.size()));
         const Eigen::VectorXd residuals = (row_sums - target).cwiseAbs();
         const bool all_normalized = (residuals.array() < tol).all();

         return all_non_negative && all_normalized;
      }
};

template <class Matrix, class Real>
struct is_left_stochastic_matrix_impl<
   Matrix, Real,
   typename std::enable_if<
      reor::backends::detail::is_eigen_matrix<Matrix>::value
      >::type> {

   static bool check(const Matrix& m, Real tol)
      {
         const bool all_non_negative = (m.array() >= 0).all();

         const Eigen::RowVectorXd col_sums = m.colwise().sum();
         const Eigen::RowVectorXd target(
            Eigen::RowVectorXd::Ones(col_sums.size()));
         const Eigen::RowVectorXd residuals = (col_sums - target).cwiseAbs();
         const bool all_normalized = (residuals.array() < tol).all();

         return all_non_negative && all_normalized;
      }
};

} // namespace detail

#endif

template <class LhsMatrix, class RhsMatrix>
bool is_equal(const LhsMatrix& a, const RhsMatrix& b,
              typename detail::common_real_type<LhsMatrix, RhsMatrix>::type
              tol = std::numeric_limits<
              typename detail::common_real_type<LhsMatrix,
              RhsMatrix>::type>::epsilon())
{
   return detail::is_equal_impl<
      LhsMatrix, RhsMatrix, decltype(tol)>::check(a, b, tol);
}

template <class Matrix>
bool is_right_stochastic_matrix(
   const Matrix& m,
   typename reor::backends::matrix_traits<Matrix>::real_element_type
   tol = std::numeric_limits<
   typename reor::backends::matrix_traits<
   Matrix>::real_element_type>::epsilon())
{
   return detail::is_right_stochastic_matrix_impl<
      Matrix, decltype(tol)>::check(m, tol);
}

template <class Matrix>
bool is_left_stochastic_matrix(
   const Matrix& m,
   typename reor::backends::matrix_traits<Matrix>::real_element_type
   tol = std::numeric_limits<
   typename reor::backends::matrix_traits<
   Matrix>::real_element_type>::epsilon())
{
   return detail::is_left_stochastic_matrix_impl<
      Matrix, decltype(tol)>::check(m, tol);
}

#endif
