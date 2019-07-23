#ifndef REOR_NUMERICS_HELPERS_HPP_INCLUDED
#define REOR_NUMERICS_HELPERS_HPP_INCLUDED

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace reor {

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
is_zero(T a, T tol = std::numeric_limits<T>::epsilon)
{
   return a < tol;
}

template <typename T>
typename std::enable_if<!std::is_unsigned<T>::value &&
                        std::is_arithmetic<T>::value, bool>::type
is_zero(T a, T tol = std::numeric_limits<T>::epsilon())
{
   using std::abs;

   return abs(a) < tol;
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, bool>::type
is_equal(T a, T b, T tol = std::numeric_limits<T>::epsilon())
{
   using std::abs;

   return abs(a - b) < tol;
}

} // namespace reor

#endif
