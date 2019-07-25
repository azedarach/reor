#ifndef REOR_CROSS_VALIDATION_HPP_INCLUDED
#define REOR_CROSS_VALIDATION_HPP_INCLUDED

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace reor {

template <class Generator>
std::vector<std::vector<int> > generate_test_sets(
   const Eigen::MatrixXd& data, int n_folds, bool oos_cv, Generator& generator,
   bool verbose)
{
   const int n_samples = data.cols();

   const auto start_time = std::chrono::high_resolution_clock::now();

   std::vector<std::vector<int> > test_sets;
   if (n_folds > 1) {
      if (oos_cv) {
         // number of folds is taken to be 1 / (fraction of data held out)
         const double test_fraction = 1. / n_folds;
         int max_training_index = static_cast<int>(
            std::floor((1 - test_fraction) * n_samples));
         if (max_training_index >= n_samples - 1) {
            max_training_index = n_samples - 2;
         }

         const int n_test_points = n_samples - 1 - max_training_index;
         std::vector<int> test_set(n_test_points);
         std::iota(std::begin(test_set), std::end(test_set),
                   max_training_index + 1);

         test_sets.push_back(test_set);
      } else {
         for (int i = 0; i < n_folds; ++i) {
            test_sets.push_back(std::vector<int>());
         }

         std::uniform_int_distribution<> dist(0, n_folds - 1);
         for (int t = 0; t < n_samples; ++t) {
            const int assignment = dist(generator);
            test_sets[assignment].push_back(t);
         }
      }
   }

   const auto end_time = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double> total_time = end_time - start_time;

   if (verbose) {
      std::cout << "Number of CV folds: " << n_folds << '\n';

      const std::size_t n_test_sets = test_sets.size();
      std::cout << "Number of test sets: " << n_test_sets << '\n';
      if (n_test_sets > 0) {
         std::cout << "Test set sizes: [";
         for (std::size_t i = 0; i < n_test_sets; ++i) {
            std::cout << test_sets[i].size();
            if (i != n_test_sets - 1) {
               std::cout << ", ";
            } else {
               std::cout << "]\n";
            }
         }
      }
      std::cout << "Required time: " << total_time.count() << "s\n";
   }

   if (test_sets.size() == 0) {
      test_sets.push_back(std::vector<int>());
   }

   return test_sets;
}

} // namespace reor

#endif
