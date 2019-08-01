#ifndef REOR_FURTHEST_SUM_HPP_INCLUDED
#define REOR_FURTHEST_SUM_HPP_INCLUDED

/**
 * @file furthest_sum.hpp
 * @brief contains implementation of FurthestSum initialization strategy
 */

#include "backend_interface.hpp"

#include <stdexcept>
#include <vector>

namespace reor {

namespace detail {

template <class Matrix, class NewIndex, class OldIndex, class Scalar>
void update_distance_sums(
   const Matrix& D, NewIndex new_index,
   std::vector<std::pair<OldIndex, Scalar> >& q)
{
   for (auto& qi : q) {
      const auto dij = backends::get_matrix_element(qi.first, new_index, D);
      qi.second += dij;
   }
}

template <class Index, class Scalar>
Index get_furthest_index(
   std::vector<std::pair<Index, Scalar> >& q)
{
   const auto comparator = [](const std::pair<Index, Scalar>& a,
                              const std::pair<Index, Scalar>& b) {
      return a.second < b.second;
   };

   std::sort(std::begin(q), std::end(q), comparator);
   const Index furthest_index = q.back().first;
   q.pop_back();

   return furthest_index;
}

template <class DissimilarityMatrix>
struct furthest_sum_impl {
   using Index = typename backends::matrix_traits<DissimilarityMatrix>::index_type;
   using Scalar = typename backends::matrix_traits<DissimilarityMatrix>::element_type;
   using Element = typename std::pair<std::size_t, Scalar>;

   template <class StartIndex, class ExcludeIndex>
   static std::vector<Index> eval(
      const DissimilarityMatrix&, std::size_t, StartIndex,
      const std::vector<ExcludeIndex>&, std::size_t);
};

template <class DissimilarityMatrix>
template <class StartIndex, class ExcludeIndex>
std::vector<typename furthest_sum_impl<DissimilarityMatrix>::Index>
furthest_sum_impl<DissimilarityMatrix>::eval(
   const DissimilarityMatrix& D, std::size_t n_components,
   StartIndex start_index, const std::vector<ExcludeIndex>& exclude,
   std::size_t extra_steps)
{
   if (n_components == 0) {
      return std::vector<Index>();
   }

   const Index n_samples = backends::cols(D);
   const Index n_excluded = exclude.size();

   if (start_index >= n_samples) {
      throw std::runtime_error(
         "starting index is out of bounds");
   }

   for (auto i : exclude) {
      if (i == start_index) {
         throw std::runtime_error(
            "starting index is excluded");
      }
   }

   if (n_excluded < n_samples &&
       static_cast<Index>(n_components) > n_samples - n_excluded) {
      throw std::runtime_error(
         "too few points to select requested number of components");
   }

   std::vector<Index> selected(n_components, start_index);

   const auto is_valid_candidate = [&selected, &exclude](Index idx) {
      bool is_valid = true;

      for (auto i : selected) {
         if (idx == i) {
            is_valid = false;
            break;
         }
      }

      for (auto i : exclude) {
         if (idx == i) {
            is_valid = false;
            break;
         }
      }

      return is_valid;
   };

   std::vector<std::pair<Index, Scalar> > q(n_samples - n_excluded - 1);
   std::size_t pos = 0;
   for (Index i = 0; i < n_samples; ++i) {
      if (is_valid_candidate(i)) {
         const auto dij = backends::get_matrix_element(i, start_index, D);
         q[pos] = std::make_pair(i, dij);
         ++pos;
      }
   }

   for (std::size_t i = 1; i < n_components; ++i) {
      selected[i] = get_furthest_index(q);
      update_distance_sums(D, selected[i], q);
   }

   if (extra_steps > 0) {
      for (std::size_t i = 0; i < extra_steps; ++i) {
         const std::size_t update_index = i % n_components;
         const Index index_to_replace = selected[update_index];

         for (auto& qi : q) {
            const auto dij = backends::get_matrix_element(
               qi.first, index_to_replace, D);
            qi.second -= dij;
         }

         // add index being replacing back to list of candidates to
         // consider
         double qi = 0;
         for (auto idx : selected) {
            if (idx != index_to_replace) {
               const auto dij = backends::get_matrix_element(
                  index_to_replace, idx, D);
               qi += dij;
            }
         }
         q.push_back(std::make_pair(index_to_replace, qi));

         selected[update_index] = get_furthest_index(q);
         update_distance_sums(D, selected[update_index], q);
      }
   }

   return selected;
}

} // namespace detail

template <class DissimilarityMatrix, class StartIndex,
          class ExcludeIndex = StartIndex>
std::vector<
   typename backends::matrix_traits<DissimilarityMatrix>::index_type>
furthest_sum(
   const DissimilarityMatrix& D, std::size_t n_components,
   StartIndex start_index,
   const std::vector<ExcludeIndex>& exclude = std::vector<ExcludeIndex>(),
   std::size_t extra_steps = 1)
{
   if (backends::rows(D) != backends::cols(D)) {
      throw std::runtime_error(
         "dissimilarity matrix must be square");
   }

   return detail::furthest_sum_impl<DissimilarityMatrix>::eval(
      D, n_components, start_index, exclude, extra_steps);
}

} // namespace reor

#endif
