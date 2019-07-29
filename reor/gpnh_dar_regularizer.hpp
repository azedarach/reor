#ifndef REOR_GPNH_DAR_REGULARIZER_HPP_INCLUDED
#define REOR_GPNH_DAR_REGULARIZER_HPP_INCLUDED

/**
 * @file gpnh_dar_regularizer.hpp
 * @brief contains definitions of GPNH-diagonal AR regularizer
 */

#include "backends.hpp"

#include <vector>

namespace reor {

/**
 * @class GPNH_DAR_regularizer
 * @brief implements GPNH and diagonal AR temporal regularization
 */
template <class Backend>
class GPNH_DAR_regularizer {
public:
   using Matrix = typename Backend::Matrix;

   void set_epsilon_states(double);
   double get_epsilon_states() const { return epsilon_S; }

   void set_epsilon_weights(double);
   double get_epsilon_weights() const { return epsilon_Gamma; }

   void set_eta_weights(double);
   double get_eta_weights() const { return eta_Gamma; }

   void set_epsilon_parameters(double);
   double get_epsilon_parameters() const { return epsilon_W; }

   const std::vector<int>& get_lag_set() const { return lag_set; }

   template <class ParametersMatrix>
   void set_parameter_values(const ParametersMatrix&);
   const Matrix& get_parameters() const { return parameters; }

   template <class WeightsMatrix>
   void initialize(const std::vector<int>&, const WeightsMatrix&);

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
   double penalty(const DataMatrix&, const StatesMatrix&,
                  const WeightsMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   void dictionary_gradient(const DataMatrix&, const StatesMatrix&,
                            const WeightsMatrix&, JacobianMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
             class JacobianMatrix>
   void weights_gradient(const DataMatrix&, const StatesMatrix&,
                              const WeightsMatrix&, JacobianMatrix&) const;

   template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
   int update_parameters(const DataMatrix&, const StatesMatrix&,
                         const WeightsMatrix&);

protected:
   ~GPNH_DAR_regularizer() = default;

private:
   double epsilon_S{0};
   double epsilon_Gamma{0};
   double eta_Gamma{0};
   double epsilon_W{};
   std::vector<int> lag_set{{1}};
   Matrix parameters;
   Matrix Z;

   void set_lag_set(const std::vector<int>&);
   int get_maximum_lag() const;

   template <class StatesMatrix>
   double states_penalty(const StatesMatrix&) const;
   template <class WeightsMatrix>
   double weights_penalty(const WeightsMatrix&) const;
   double parameters_penalty() const;
};

template <class Backend>
void GPNH_DAR_regularizer<Backend>::set_epsilon_states(double eps)
{
   if (eps < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   epsilon_S = eps;
}

template <class Backend>
void GPNH_DAR_regularizer<Backend>::set_epsilon_weights(double eps)
{
   if (eps < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   epsilon_Gamma = eps;
}

template <class Backend>
void GPNH_DAR_regularizer<Backend>::set_eta_weights(double eta)
{
   if (eta < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   eta_Gamma = eta;
}

template <class Backend>
void GPNH_DAR_regularizer<Backend>::set_epsilon_parameters(double eps)
{
   if (eps < 0) {
      throw std::runtime_error("regularization parameter must be non-negative");
   }

   epsilon_W = eps;
}

template <class Backend>
void GPNH_DAR_regularizer<Backend>::set_lag_set(
   const std::vector<int>& lags)
{
   if (lags.empty()) {
      throw std::runtime_error("lag set must be non-empty");
   }

   for (auto l : lags) {
      if (l < 1) {
         throw std::runtime_error("lags must be at least one");
      }
   }

   lag_set = lags;
}

template <class Backend>
template <class WeightsMatrix>
void GPNH_DAR_regularizer<Backend>::initialize(
   const std::vector<int>& lags, const WeightsMatrix& Gamma)
{
   set_lag_set(lags);

   const std::size_t n_lags = lag_set.size();
   const std::size_t n_components = backends::rows(Gamma);
   const std::size_t n_samples = backends::cols(Gamma);
   const int max_lag = get_maximum_lag();

   parameters = Backend::create_constant_matrix(
      n_lags, n_components, 1);
   Z = Backend::create_matrix(n_lags, n_samples - max_lag);
}

template <class Backend>
template <class ParametersMatrix>
void GPNH_DAR_regularizer<Backend>::set_parameter_values(
   const ParametersMatrix& parameters_)
{
   const std::size_t n_lags = lag_set.size();

   if (backends::rows(parameters_) != n_lags) {
      throw std::runtime_error(
         "number of rows does not match number of lags");
   }

   const std::size_t n_components = backends::cols(parameters);

   if (backends::cols(parameters_) != n_components) {
      throw std::runtime_error(
         "number of columns does not match number of components");
   }

   parameters = Backend::copy_matrix(parameters_);
}

template <class Backend>
int GPNH_DAR_regularizer<Backend>::get_maximum_lag() const
{
   int max_lag = -1;
   for (auto l : lag_set) {
      if (l > max_lag) {
         max_lag = l;
      }
   }
   return max_lag;
}

template <class Backend>
template <class StatesMatrix>
double GPNH_DAR_regularizer<Backend>::states_penalty(
   const StatesMatrix& S) const
{
   double value = 0;

   const std::size_t n_components = backends::cols(S);
   const std::size_t n_features = backends::rows(S);

   if (n_components == 1) {
      return value;
   }

   const double prefactor = 2.0 / (n_components * n_features *
                                   (n_components - 1));
   value += prefactor * n_components * backends::trace_gemm_op(
      1, S, S, backends::Op_flag::Transpose)
      - prefactor * backends::sum_gemm_op(
         1, S, S, backends::Op_flag::Transpose, backends::Op_flag::None);

   return epsilon_S * value;
}

template <class Backend>
template <class WeightsMatrix>
double GPNH_DAR_regularizer<Backend>::weights_penalty(
   const WeightsMatrix& Gamma) const
{
   double tr_penalty = 0;

   const std::size_t n_components = backends::rows(Gamma);
   const std::size_t n_samples = backends::cols(Gamma);
   const std::size_t n_lags = lag_set.size();
   const std::size_t max_lag = get_maximum_lag();

   for (std::size_t t = max_lag; t < n_samples; ++t) {
      for (std::size_t i = 0; i < n_components; ++i) {
         double r = backends::get_matrix_element(i, t, Gamma);
         for (std::size_t l = 0; l < n_lags; ++l) {
            r -= backends::get_matrix_element(l, i, parameters) *
               backends::get_matrix_element(i, t - lag_set[l], Gamma);
         }
         tr_penalty += 0.5 * r * r;
      }
   }

   const double gamma_norm = backends::matrix_fro_norm(Gamma);
   const double norm_penalty = 0.5 * gamma_norm * gamma_norm;

   return epsilon_Gamma * tr_penalty + eta_Gamma * norm_penalty;
}

template <class Backend>
double GPNH_DAR_regularizer<Backend>::parameters_penalty() const
{
   const double parameters_norm = backends::matrix_fro_norm(parameters);
   const double norm_penalty = parameters_norm * parameters_norm;

   return epsilon_W * norm_penalty;
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
double GPNH_DAR_regularizer<Backend>::penalty(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const WeightsMatrix& Gamma) const
{
   return states_penalty(S) + weights_penalty(Gamma)
      + parameters_penalty();
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void GPNH_DAR_regularizer<Backend>::dictionary_gradient(
   const DataMatrix& /* X */, const StatesMatrix& S,
   const WeightsMatrix& /* Gamma */, JacobianMatrix& jac_S) const
{
   const std::size_t n_components = backends::cols(S);
   const std::size_t n_features = backends::rows(S);

   if (n_components == 1) {
      for (std::size_t i = 0; i < n_features; ++i) {
         for (std::size_t j = 0; j < n_components; ++j) {
            backends::set_matrix_element(i, j, 0, jac_S);
         }
      }
   } else {
      const double prefactor = 4.0 * epsilon_S /
         (n_features * n_components * (n_components - 1));

      auto D = Backend::create_diagonal_matrix(
         n_components, n_components);
      backends::add_constant(-1, D);

      backends::gemm(prefactor, S, D, 0, jac_S);
   }
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix,
          class JacobianMatrix>
void GPNH_DAR_regularizer<Backend>::weights_gradient(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& Gamma, JacobianMatrix& jac_Gamma) const
{
   const std::size_t n_components = backends::rows(Gamma);
   const std::size_t n_samples = backends::cols(Gamma);
   const std::size_t n_lags = lag_set.size();
   const std::size_t max_lag = get_maximum_lag();

   for (std::size_t t = 0; t < n_samples; ++t) {
      for (std::size_t i = 0; i < n_components; ++i) {
         const auto gamma_it = backends::get_matrix_element(i, t, Gamma);

         double value = eta_Gamma * gamma_it;

         if (t >= max_lag) {
            double r = gamma_it;
            for (std::size_t l = 0; l < n_lags; ++l) {
               r -= backends::get_matrix_element(i, l, parameters)
                  * backends::get_matrix_element(i, t - lag_set[l], Gamma);
            }
            value += epsilon_Gamma * r;
         }

         for (std::size_t l = 0; l < n_lags; ++l) {
            if (t + lag_set[l] < n_samples) {
               double r = backends::get_matrix_element(i, t + lag_set[l], Gamma);
               for (std::size_t lp = 0; lp < n_lags; ++lp) {
                  r -= backends::get_matrix_element(i, lp, parameters)
                     * backends::get_matrix_element(i, t + lag_set[l] - lag_set[lp],
                                                    Gamma);
               }
               value -= epsilon_Gamma * backends::get_matrix_element(
                  i, l, parameters) * r;
            }
         }

         backends::set_matrix_element(i, t, value, jac_Gamma);
      }
   }
}

template <class Backend>
template <class DataMatrix, class StatesMatrix, class WeightsMatrix>
int GPNH_DAR_regularizer<Backend>::update_parameters(
   const DataMatrix& /* X */, const StatesMatrix& /* S */,
   const WeightsMatrix& Gamma)
{
   const std::size_t n_components = backends::rows(Gamma);
   const std::size_t n_samples = backends::cols(Gamma);
   const std::size_t n_lags = lag_set.size();
   const std::size_t max_lag = get_maximum_lag();

   Matrix ZZt = Backend::create_diagonal_matrix(
      n_lags, epsilon_W / epsilon_Gamma);
   Matrix ZGt = Backend::create_matrix(n_lags, 1);

   int status = 0;
   for (std::size_t i = 0; i < n_components; ++i) {

      for (std::size_t l = 0; l < n_lags; ++l) {
         double rhs_elem = 0;
         for (std::size_t t = max_lag; t < n_samples; ++t) {
            const auto zlt =
               backends::get_matrix_element(i, t - lag_set[l], Gamma);
            backends::set_matrix_element(
               l, t, zlt, Z);
            rhs_elem += zlt * backends::get_matrix_element(i, t, Gamma);
         }
         backends::set_matrix_element(l, 1, rhs_elem, ZGt);
      }

      backends::gemm(1, Z, Z, 1, ZZt, backends::Op_flag::None,
                     backends::Op_flag::Transpose);

      const auto error = backends::solve_ldlt(ZZt, ZGt);
      if (error) {
         status = error;
      }

      for (std::size_t l = 0; l < n_lags; ++l) {
         const auto w = backends::get_matrix_element(l, 1, ZGt);
         backends::set_matrix_element(l, i, w, parameters);
      }
   }

   return status;
}

} // namespace reor

#endif
