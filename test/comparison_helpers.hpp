#ifndef REOR_COMPARISON_HELPERS_HPP_INCLUDED
#define REOR_COMPARISON_HELPERS_HPP_INCLUDED

#include <limits>
#include <type_traits>

namespace detail {

template <class LhsMatrix, class RhsMatrix, class Enable = void>
struct common_real_type {};

template <class LhsMatrix, class RhsMatrix, class Scalar, class Enable = void>
struct is_equal_impl {};

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

#endif
