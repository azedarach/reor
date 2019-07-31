#ifndef REOR_PYREOR_EIGEN_L2_SPA_HPP_INCLUDED
#define REOR_PYREOR_EIGEN_L2_SPA_HPP_INCLUDED

/**
 * @file pyreor_eigen_l2_spa.hpp
 * @brief contains definition of wrapper class for SPA factorization
 */

#include "reor/backends/eigen_backend.hpp"
#include "reor/gpnh_regularizer.hpp"
#include "reor/l2_spa.hpp"

#include <Eigen/Core>

namespace reor {

class EigenL2SPAGPNH {
public:
   EigenL2SPAGPNH(
      const Eigen::Ref<const Eigen::MatrixXd>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& dictionary,
      const Eigen::Ref<const Eigen::MatrixXd>& weights)
      : spa(data, dictionary, weights)
      {
      }
   ~EigenL2SPAGPNH() = default;

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
   using Regularization = GPNH_regularizer<Backend>;

   L2_SPA<Backend, Regularization> spa;
};

} // namespace reor

#endif
