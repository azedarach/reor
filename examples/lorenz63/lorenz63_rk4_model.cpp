#include "lorenz63_rk4_model.hpp"

namespace reor {

void Lorenz63_rk4_model::step()
{
   const double t = 0;
   std::array<double, 3> dxdt;

   system(state, dxdt, t);
   std::array<double, 3> k1;
   for (std::size_t i = 0; i < 3; ++i) {
      k1[i] = time_step * dxdt[i];
   }

   std::array<double, 3> xphk1(state);
   for (std::size_t i = 0; i < 3; ++i) {
      xphk1[i] += 0.5 * k1[i];
   }
   system(xphk1, dxdt, t);
   std::array<double, 3> k2;
   for (std::size_t i = 0; i < 3; ++i) {
      k2[i] = time_step * dxdt[i];
   }

   std::array<double, 3> xphk2(state);
   for (std::size_t i = 0; i < 3; ++i) {
      xphk2[i] += 0.5 * k2[i];
   }
   system(xphk2, dxdt, t);
   std::array<double, 3> k3;
   for (std::size_t i = 0; i < 3; ++i) {
      k3[i] = time_step * dxdt[i];
   }

   std::array<double, 3> xpk3(state);
   for (std::size_t i = 0; i < 3; ++i) {
      xpk3[i] += k3[i];
   }
   system(xpk3, dxdt, t);
   std::array<double, 3> k4;
   for (std::size_t i = 0; i < 3; ++i) {
      k4[i] = time_step * dxdt[i];
   }

   for (std::size_t i = 0; i < 3; ++i) {
      state[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
   }
}

} // namespace reor

