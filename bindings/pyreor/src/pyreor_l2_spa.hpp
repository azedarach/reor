#ifndef REOR_PYREOR_L2_SPA_HPP_INCLUDED
#define REOR_PYREOR_L2_SPA_HPP_INCLUDED

/**
 * @file pyreor_l2_spa.hpp
 * @brief contains definitions of wrapper classes for SPA factorization
 */

#include "reor/l2_spa.hpp"

#include <pybind11/pybind11.h>

class BuiltinL2SPAGPNH {
public:
   BuiltinL2SPAGPNH();
   ~BuiltinL2SPAGPNH();
};

#ifdef HAVE_EIGEN
#include "pyreor_eigen_l2_spa.hpp"
#endif

#endif
