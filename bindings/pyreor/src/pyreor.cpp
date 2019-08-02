/**
 * @file pyreor.cpp
 * @brief provides Python binding definitions
 */

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include "pyreor_kernel_aa.hpp"
#include "pyreor_l2_spa.hpp"

namespace py = pybind11;

using namespace reor;

py::tuple get_available_backends()
{
   return py::make_tuple(
      "builtin"
#ifdef HAVE_EIGEN
      , "eigen"
#endif
      );
}

PYBIND11_MODULE(pyreor_ext, m) {
   m.def("backends", &get_available_backends,
         "Return tuple of available backend names");

#ifdef HAVE_EIGEN
   py::class_<EigenL2SPAGPNH>(m, "EigenL2SPAGPNH")
      .def(py::init<
           const Eigen::Ref<const Eigen::MatrixXd>&,
           const Eigen::Ref<const Eigen::MatrixXd>&,
           const Eigen::Ref<const Eigen::MatrixXd>& >())
      .def_property("epsilon_states",
                    &EigenL2SPAGPNH::get_epsilon_states,
                    &EigenL2SPAGPNH::set_epsilon_states)
      .def("get_dictionary", &EigenL2SPAGPNH::get_dictionary,
           py::return_value_policy::copy)
      .def("get_weights", &EigenL2SPAGPNH::get_weights,
           py::return_value_policy::copy)
      .def("cost", &EigenL2SPAGPNH::cost)
      .def("update_dictionary", &EigenL2SPAGPNH::update_dictionary)
      .def("update_weights", &EigenL2SPAGPNH::update_weights);

   m.def("furthest_sum_eigen", &furthest_sum_eigen,
         py::arg("dissimilarities"),
         py::arg("n_components"),
         py::arg("starting_index"),
         py::arg("exclude") = std::vector<Eigen::Index>(),
         py::arg("extra_steps") = 10);

   py::class_<EigenKernelAA>(m, "EigenKernelAA")
      .def(py::init<
           const Eigen::Ref<const Eigen::MatrixXd>&,
           const Eigen::Ref<const Eigen::MatrixXd>&,
           const Eigen::Ref<const Eigen::MatrixXd>&,
           double>())
      .def("get_dictionary", &EigenKernelAA::get_dictionary,
           py::return_value_policy::copy)
      .def("get_weights", &EigenKernelAA::get_weights,
           py::return_value_policy::copy)
      .def("get_scale_factors", &EigenKernelAA::get_scale_factors,
           py::return_value_policy::copy)
      .def("cost", &EigenKernelAA::cost)
      .def("update_dictionary", &EigenKernelAA::update_dictionary)
      .def("update_weights", &EigenKernelAA::update_weights);
#endif
}
