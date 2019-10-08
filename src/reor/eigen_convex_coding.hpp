#ifndef REOR_EIGEN_CONVEX_CODING_HPP_INCLUDED
#define REOR_EIGEN_CONVEX_CODING_HPP_INCLUDED

/**
 * @file eigen_convex_coding.hpp
 * @brief contains definition of wrapper class for convex coding routines
 */

#include "reor_cxx/backends/eigen_backend.hpp"
#include "reor_cxx/gpnh_l2_spa.hpp"

#include <Eigen/Core>

namespace reor {

class EigenGPNHL2SPA {
public:
   EigenGPNHL2SPA(
      const Eigen::Ref<const Eigen::MatrixXd>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& dictionary,
      const Eigen::Ref<const Eigen::MatrixXd>& weights)
      : spa(data, dictionary, weights)
      {
      }
   ~EigenGPNHL2SPA() = default;

   void set_epsilon_states(double eps) { spa.set_epsilon_states(eps); }
   double get_epsilon_states() const { return spa.get_epsilon_states(); }

   const Eigen::MatrixXd& get_dictionary() const {
      return spa.get_dictionary();
   }
   const Eigen::MatrixXd& get_weights() const { return spa.get_weights(); }

   double cost() const { return spa.cost(); }

   int update_dictionary() { return spa.update_dictionary(); }
   int update_weights() { return spa.update_weights(); }

private:
   using Backend = backends::Eigen_backend<double>;

   GPNH_L2_SPA<Backend> spa;
};

} // namespace reor

#endif
