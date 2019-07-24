#ifndef REOR_LORENZ63_SYSTEM_HPP_INCLUDED
#define REOR_LORENZ63_SYSTEM_HPP_INCLUDED

#include <array>

namespace reor {

struct Lorenz63_system {
   double rho{28};
   double sigma{10};
   double beta{8. / 3.};

   template <class State, class Deriv>
   void operator()(const State&, Deriv&, double) const;
   template <class State>
   State operator()(const State&, double) const;
};

template <class State, class Deriv>
void Lorenz63_system::operator()(const State& x, Deriv& dxdt, double /* t */) const
{
   dxdt[0] = sigma * (x[1] - x[0]);
   dxdt[1] = rho * x[0] - x[1] - x[0] * x[2];
   dxdt[2] = x[0] * x[1] - beta * x[2];
}

template <class State>
State Lorenz63_system::operator()(const State& x, double t) const
{
   State dxdt = x;
   operator()(x, dxdt, t);
   return dxdt;
}

} // namespace reor

#endif
