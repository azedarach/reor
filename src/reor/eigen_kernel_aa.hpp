#ifndef REOR_EIGEN_KERNEL_AA_HPP_INCLUDED
#define REOR_EIGEN_KERNEL_AA_HPP_INCLUDED

/**
 * @file eigen_kernel_aa.hpp
 * @brief contains definition of wrapper class for kernel AA
 */

#include "reor_cxx/backends/eigen_backend.hpp"
#include "reor_cxx/furthest_sum.hpp"
#include "reor_cxx/kernel_aa.hpp"

#include <Eigen/Core>

namespace reor {

std::vector<Eigen::Index> furthest_sum_eigen(
   const Eigen::Ref<const Eigen::MatrixXd>& dissimilarities,
   std::size_t n_components, Eigen::Index starting_index,
   const std::vector<Eigen::Index> exclude, std::size_t extra_steps)
{
   return furthest_sum(dissimilarities, n_components, starting_index,
                       exclude, extra_steps);
}

class EigenKernelAA {
public:
   EigenKernelAA(
      const Eigen::Ref<const Eigen::MatrixXd>& kernel,
      const Eigen::Ref<const Eigen::MatrixXd>& dictionary,
      const Eigen::Ref<const Eigen::MatrixXd>& weights,
      double delta)
      : aa(kernel, dictionary, weights, delta)
      {
      }
   ~EigenKernelAA() = default;

   const Eigen::MatrixXd& get_dictionary() const {
      return aa.get_dictionary();
   }
   const Eigen::MatrixXd& get_weights() const {
      return aa.get_weights();
   }
   const Eigen::MatrixXd& get_scale_factors() const {
      return aa.get_scale_factors();
   }

   double cost() { return aa.cost(); }

   int update_dictionary() { return aa.update_dictionary(); }
   int update_weights() { return aa.update_weights(); }

private:
   using Backend = backends::Eigen_backend<double>;

   KernelAA<Backend> aa;
};

} // namespace reor

#endif
