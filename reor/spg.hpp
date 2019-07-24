#ifndef REOR_SPG_HPP_INCLUDED
#define REOR_SPG_HPP_INCLUDED

namespace reor {

template <typename T = double>
struct SPG_line_search_parameters {
   T alpha_min{1e-3};
   T alpha_max{10};
   T sigma_1{0.1};
   T sigma_2{0.9};
   T gamma{1e-4};

   template <typename T1, typename T2, typename T3, typename T4>
   T1 get_next_step_length(T1 lambda, T2 delta, T3 f_old, T4 f_new) const;
   template <typename T1, typename T2>
   T get_next_alpha(T1 beta, T2 sksk) const;
};

template <typename T>
template <typename T1, typename T2, typename T3, typename T4>
T1 SPG_line_search_parameters<T>::get_next_step_length(
   T1 lambda, T2 delta, T3 f_old, T4 f_new) const
{
   const auto lambda_tmp = -0.5 * lambda * lambda * delta /
      (f_new - f_old - lambda * delta);

   T1 next_lambda = 0;
   if (lambda_tmp >= sigma_1 && lambda_tmp <= sigma_2 * lambda) {
      next_lambda = lambda_tmp;
   } else {
      next_lambda = 0.5 * lambda;
   }

   return next_lambda;
}

template <typename T>
template <typename T1, typename T2>
T SPG_line_search_parameters<T>::get_next_alpha(T1 beta, T2 sksk) const
{
   using std::min;
   using std::max;

   if (beta <= 0) {
      return alpha_max;
   }

   return min(alpha_max, max(alpha_min, sksk / beta));
}

} // namespace reor

#endif
