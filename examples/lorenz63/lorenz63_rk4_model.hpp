#ifndef REOR_LORENZ63_RK4_MODEL_HPP_INCLUDED
#define REOR_LORENZ63_RK4_MODEL_HPP_INCLUDED

#include "lorenz63_system.hpp"

#include <array>

namespace reor {

struct Lorenz63_rk4_model {
   double time_step{1e-3};
   std::array<double, 3> state{{0, 0, 0}};
   Lorenz63_system system{};

   void step();
};

} // namespace reor

#endif
