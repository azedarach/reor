#ifndef REOR_EIGEN_TYPE_TRAITS_HPP_INCLUDED
#define REOR_EIGEN_TYPE_TRAITS_HPP_INCLUDED

/**
 * @file eigen_type_traits.hpp
 * @brief helper type inspection routines for Eigen types
 */

#include <Eigen/Core>

#include <type_traits>

namespace reor {

namespace backends {

namespace detail {

template <class T, class Enable = void>
struct is_eigen_matrix : public std::false_type {};

template <class Matrix>
struct is_eigen_matrix<
   Matrix,
   typename std::enable_if<
      std::is_base_of<
         Eigen::MatrixBase<Matrix>,
         Matrix>::value>::type>
   : public std::true_type {};

} // namespace detail

} // namespace backends

} // namespace reor

#endif
