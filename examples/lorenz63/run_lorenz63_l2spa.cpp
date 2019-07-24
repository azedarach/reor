#include "lorenz63_rk4_model.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

using namespace reor;

struct Program_options {
   int n_components{2};
   int n_folds{1};
   int n_init{10};
   int n_samples{5000};
   int random_seed{0};
   double x0{1};
   double y0{1};
   double z0{1};
   double beta{8. / 3.};
   double sigma{10};
   double rho{28};
   double time_step{1e-2};
   std::string trajectory_output_file{""};
   bool out_of_sample_cv{false};
   bool verbose{false};
};

void print_usage()
{
   std::cout <<
      "Usage: run_lorenz63_l2spa [OPTION]\n\n"
      "Calculate l2-SPA factorizations for Lorenz-63 trajectory.\n\n"
      "Example: run_lorenz63_l2spa -T 5000\n\n"
      "Options:\n"
      "  --beta=BETA                         value of system parameter beta\n"
      "  -d, --trajectory-output-file=FILE   file to write trajectory data to\n"
      "  -h, --help                          print this help message\n"
      "  --rho=RHO                           value of system parameter rho\n"
      "  --sigma=SIGMA                       value of system parameter sigma\n"
      "  -t, --time=step=TIME_STEP           time step\n"
      "  -T, --n-samples=N_SAMPLES           number of samples\n"
      "  -v, --verbose                       produce verbose output\n"
      "  --x0=X0                             initial value for x\n"
      "  --y0=Y0                             initial value for y\n"
      "  --z0=Z0                             initial value for z\n"
             << std::endl;
}

bool starts_with(const std::string& option, const std::string& prefix)
{
   return !option.compare(0, prefix.size(), prefix);
}

std::string get_option_value(const std::string& option,
                             const std::string& sep = "=")
{
   std::string value{""};
   const auto prefix_end = option.find(sep);

   if (prefix_end != std::string::npos) {
      value = option.substr(prefix_end + 1);
   }

   return value;
}

Program_options parse_cmd_line_args(int argc, const char* argv[])
{
   Program_options options;

   int i = 1;
   while (i < argc) {
      const std::string opt(argv[i++]);

      if (opt == "-d") {
         if (i == argc) {
            throw std::runtime_error(
               "'-d' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-d' given but no output file name provided");
         }
         options.trajectory_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--trajectory-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--trajectory-output-file=' given but no output file name provided");
         }
         options.trajectory_output_file = filename;
         continue;
      }

      if (opt == "-h" || opt == "--help") {
         print_usage();
         exit(EXIT_SUCCESS);
      }

      if (opt == "-t") {
         if (i == argc) {
            throw std::runtime_error(
               "'-t' given but time step not provided");
         }
         const std::string time_step(argv[i++]);
         if (starts_with(time_step, "-")) {
            throw std::runtime_error(
               "'-t' given but valid time step not provided");
         }
         options.time_step = std::stod(time_step);
         continue;
      }

      if (starts_with(opt, "--time-step=")) {
         const std::string time_step = get_option_value(opt);
         if (time_step.empty()) {
            throw std::runtime_error(
               "'--time-step=' given but no time step provided");
         }
         options.time_step = std::stod(time_step);
         continue;
      }

      if (opt == "-T") {
         if (i == argc) {
            throw std::runtime_error(
               "'-T' given but number of samples not provided");
         }
         const std::string n_samples(argv[i++]);
         if (starts_with(n_samples, "-")) {
            throw std::runtime_error(
               "'-T' given but valid number of samples not provided");
         }
         options.n_samples = std::stoi(n_samples);
         continue;
      }

      if (starts_with(opt, "--n-samples=")) {
         const std::string n_samples = get_option_value(opt);
         if (n_samples.empty() || starts_with(n_samples, "-")) {
            throw std::runtime_error(
               "'--n-samples=' given but valid number of samples not provided");
         }
         options.n_samples = std::stoi(n_samples);
         continue;
      }

      if (starts_with(opt, "--beta=")) {
         const std::string beta = get_option_value(opt);
         if (beta.empty()) {
            throw std::runtime_error(
               "'--beta=' given but no parameter value provided");
         }
         options.beta = std::stod(beta);
         continue;
      }

      if (starts_with(opt, "--rho=")) {
         const std::string rho = get_option_value(opt);
         if (rho.empty()) {
            throw std::runtime_error(
               "'--rho=' given but no parameter value provided");
         }
         options.rho = std::stod(rho);
         continue;
      }

      if (starts_with(opt, "--sigma=")) {
         const std::string sigma = get_option_value(opt);
         if (sigma.empty()) {
            throw std::runtime_error(
               "'--sigma=' given but no parameter value provided");
         }
         options.sigma = std::stod(sigma);
         continue;
      }

      if (opt == "-v" || opt == "--verbose") {
         options.verbose = true;
         continue;
      }

      if (starts_with(opt, "--x0=")) {
         const std::string x0 = get_option_value(opt);
         if (x0.empty()) {
            throw std::runtime_error(
               "'--x0=' given but no initial value provided");
         }
         options.x0 = std::stod(x0);
         continue;
      }

      if (starts_with(opt, "--y0=")) {
         const std::string y0 = get_option_value(opt);
         if (y0.empty()) {
            throw std::runtime_error(
               "'--y0=' given but no initial value provided");
         }
         options.y0 = std::stod(y0);
         continue;
      }

      if (starts_with(opt, "--z0=")) {
         const std::string z0 = get_option_value(opt);
         if (z0.empty()) {
            throw std::runtime_error(
               "'--z0=' given but no initial value provided");
         }
         options.z0 = std::stod(z0);
         continue;
      }
   }

   return options;
}

void generate_trajectory(
   Lorenz63_rk4_model model, int n_samples,
   Eigen::VectorXd& times, Eigen::MatrixXd& data, bool verbose)
{
   if (verbose) {
      std::cout << "System parameter values:\n";
      std::cout << "\tbeta = " << model.system.beta << '\n';
      std::cout << "\tsigma = " << model.system.sigma << '\n';
      std::cout << "\trho = " << model.system.rho << '\n';
      std::cout << "Initial values:\n";
      std::cout << "\tx0 = " << model.state[0] << '\n';
      std::cout << "\ty0 = " << model.state[1] << '\n';
      std::cout << "\tz0 = " << model.state[2] << '\n';
      std::cout << "Generating input trajectory ...";
   }

   const auto gen_start_time = std::chrono::high_resolution_clock::now();

   times = Eigen::VectorXd::Zero(n_samples);
   data = Eigen::MatrixXd::Zero(3, n_samples);

   for (int i = 0; i < 3; ++i) {
      data(i, 0) = model.state[i];
   }

   for (int t = 1; t < n_samples; ++t) {
      times(t) = times(t - 1) + model.time_step;
      model.step();
      for (int i = 0; i < 3; ++i) {
         data(i, t) = model.state[i];
      }
   }

   const auto gen_end_time = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double> gen_time = gen_end_time - gen_start_time;

   if (verbose) {
      std::cout << " done [" << gen_time.count() << "s]\n";
   }
}

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
   const std::duration<double> total_time = end_time - start_time;

   if (verbose) {
      std::cout << "Number of CV folds: " << n_folds << '\n';

      const std::size_t n_test_sets = test_sets.size();
      std::cout << "Number of test sets: " << n_test_sets << '\n';
      std::cout << "Test set sizes: [";
      for (std::size_t i = 0; i < n_test_sets; ++i) {
         std::cout << test_sets[i].size();
         if (i != n_test_sets - 1) {
            std::cout << ", ";
         } else {
            std::cout << "]\n";
         }
      }
      std::cout << "Required time: " << total_time.count() << "s\n";
   }

   return test_sets;
}

double calculate_rmse(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
   const Eigen::MatrixXd residuals(A - B);
   const double mse = residuals.cwiseProduct(residuals).mean();
   return std::sqrt(mse);
}

struct Factorization_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd affiliations{};
   double min_cost{-1};
   double max_cost{-1};
   double average_cost{-1};
   double min_training_approx_rmse{-1};
   double max_training_approx_rmse{-1};
   double average_training_approx_rmse{-1};
   double min_test_approx_rmse{-1};
   double max_test_approx_rmse{-1};
   double average_test_approx_rmse{-1};
   double min_time_seconds{-1};
   double max_time_seconds{-1};
   double average_time_seconds{-1};
};

struct Fit_result {
   Eigen::MatrixXd dictionary{};
   Eigen::MatrixXd affiliations{};
   double cost{std::numeric_limits<double>::max()};
   double training_rmse{std::numeric_limits<double>::max()};
   double test_rmse{std::numeric_limits<double>::max()};
   int n_iter{-1};
   double time_seconds{-1};
   bool success{false};
};

std::tuple<bool, int, double> iterate_until_converged(
   L2_SPA<Backend, Regularization>& spa, double tolerance, int max_iterations)
{
   double old_cost = spa.cost();
   double new_cost = old_cost;
   double cost_delta = std::numeric_limits<double>::max();
   bool success = false;

   int iter = 0;
   while (iter < max_iterations) {
      old_cost = new_cost;

      spa.update_dictionary();

      const double tmp_cost = spa.cost();
      if (tmp_cost > old_cost) {
         throw std::runtime_error(
            "factorization cost increased after dictionary update");
      }

      spa.update_affiliations();

      new_cost = spa.cost();
      cost_delta = new_cost - old_cost;

      if (cost_delta > 0) {
         throw std::runtime_error(
            "factorization cost increased after affiliations update");
      }

      if (std::abs(cost_delta) < tolerance) {
         success = true;
         break;
      }

      ++iter;
   }

   return std::make_tuple(success, iter, new_cost);
}

template <class Generator>
Fit_result run_l2spa(
   const Eigen::MatrixXd& data, const std::vector<int>& test_set,
   int n_components, double epsilon_states, int n_init,
   double tolerance, int max_iterations, Generator& generator)
{
   Fit_result best_result;

   const int n_features = data.rows();
   const int n_samples = data.cols();
   const int n_training_samples = n_samples - test_set.size();

   Eigen::MatrixXd training_data(
      Eigen::MatrixXd::Zero(n_features, n_training_samples));
   int training_idx = 0;
   for (int i = 0; i < n_samples; ++i) {
      const bool in_test_set = std::find(
         std::begin(test_set), std::end(test_set), i) == std::end(test_set);

      if (!in_test_set) {
         training_data.col(training_idx) = data.col(i);
         ++training_idx;
      }
   }

   for (int i = 0; i < n_init; ++i) {
      const auto start_time = std::chrono::high_resolution_clock::now();

      L2_SPA<Backend, Regularization> spa(
         training_data, initial_dictionary, initial_affiliations);
      spa.set_epsilon_states(epsilon_states);

      const std::tuple<bool, int, double> iteration_result =
         iterate_until_converged(spa, tolerance, max_iterations);

      const bool success = std::get<0>(iteration_result);
      const int n_iter = std::get<1>(iteration_result);
      const double cost = std::get<2>(iteration_result);

      const auto end_time = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> total_time = end_time - start_time;

      if (success && new_cost < best_result.cost) {
         best_result.dictionary = spa.get_dictionary();
         best_result.affiliations = spa.get_affiliations();
         best_result.cost = new_cost;
         best_result.n_iter = iter;
         best_result.time_seconds = total_time.count();
         best_result.success = success;
      }
   }

   if (best_result.success) {
      const Eigen::MatrixXd reconstruction =
         best_result.dictionary * best_result.affiliations;
      best_result.training_rmse = calculate_rmse(training_data, reconstruction);

      if (test_set.size() != 0) {
         const int n_test_samples = test_set.size();
         Eigen::MatrixXd test_data(n_features, n_test_samples);
         for (int i = 0; i < n_test_samples; ++i) {
            test_data.col(i) = data.col(test_set[i]);
         }

         L2_SPA<Backend, Regularization> spa(
            test_data, best_result.dictionary, initial_affiliations);

      }
   }

   return best_result;
}

Factorization_result run_cross_validated_l2spa(
   const Eigen::MatrixXd& data, const std::vector<std::vector<int> >& test_sets,
   int n_components, double epsilon_states, int n_init)
{

}

std::vector<Factorization_result> calculate_factorization(
   const Eigen::MatrixXd& data, const std::vector<int>& n_components,
   const std::vector<double>& epsilon_states,
   int n_folds, int n_init, int random_seed, bool oos_cv, bool verbose)
{
   if (verbose) {
      std::cout << "Running factorization algorithm\n";
      std::cout << "Random seed: " << random_seed << '\n';
   }

   const auto start_time = std::chrono::high_resolution_clock::now();

   std::mt19937 generator(random_seed);

   if (verbose) {
      std::cout << "Generating cross-validation test sets\n";
   }

   std::vector<std::vector<int> > test_sets = generate_test_sets(
      data, n_folds, oos_cv, generator, verbose);

   std::vector<Factorization_result> results;
   for (int k : n_components) {
      for (double eps : epsilon_states) {
         Factorization_result result = run_cross_validated_l2spa(
            data, test_sets, k, epsilon_states, n_init, initial_dictionary,
            initial_affiliations, generator, verbose);

         results.push_back(result);
      }
   }

   const auto end_time = std::chrono::high_resolution_clock::now();
   const std::chrono::duration<double> total_time = end_time - start_time;

   if (verbose) {
      std::cout << "Total time: " << total_time.count() << "s\n";
   }

   return results;
}

void write_header_line(
   std::ofstream& ofs, const std::vector<std::string>& fields)
{
   const std::size_t n_fields = fields.size();

   std::string header = "# ";
   for (std::size_t f = 0; f < n_fields; ++f) {
      header = header + fields[f];
      if (f != n_fields - 1) {
         header = header + ',';
      }
   }
   header = header + '\n';

   ofs << header;
}

void write_data_lines(std::ofstream& ofs, const Eigen::MatrixXd& data)
{
   const int n_fields = data.rows();
   const int n_samples = data.cols();

   for (int t = 0; t < n_samples; ++t) {
      std::stringstream sstr;
      for (int i = 0; i < n_fields; ++i) {
         sstr << std::scientific << std::setw(18) << data(i, t);
         if (i != n_fields - 1) {
            sstr << ',';
         }
      }
      sstr << '\n';
      ofs << sstr.str();
   }
}

void write_csv(const std::string& output_file, const Eigen::MatrixXd& data,
               const std::vector<std::string>& fields)
{
   const int n_fields = fields.size();
   if (data.rows() != n_fields) {
      throw std::runtime_error(
         "number of data rows does not match number of fields");
   }

   std::ofstream ofs(output_file);

   if (!ofs.is_open()) {
      throw std::runtime_error(
         "failed to open datafile for writing");
   }

   write_header_line(ofs, fields);
   write_data_lines(ofs, data);
}

void write_trajectory_csv(
   const std::string& output_file,
   const Eigen::VectorXd& times, const Eigen::MatrixXd& states)
{
   const int n_samples = times.size();
   const int n_features = states.rows();

   std::vector<std::string> fields(n_features + 1);
   fields[0] = "t";
   for (int i = 1; i <= n_features; ++i) {
      fields[i] = "x" + std::to_string(i);
   }

   Eigen::MatrixXd data(n_features + 1, n_samples);
   data.row(0) = times;
   data.block(1, 0, n_features, n_samples) = states;

   write_csv(output_file, data, fields);
}

int main(int argc, const char* argv[])
{
   try {
      const auto args = parse_cmd_line_args(argc, argv);

      if (args.n_samples < 1) {
         std::cerr << "Error: number of samples must be at least one."
                   << std::endl;
         return 1;
      }

      if (args.time_step == 0) {
         std::cerr << "Error: time step must be non-zero."
                   << std::endl;
         return 1;
      }

      const int n_samples = args.n_samples;
      const int n_features = 3;

      std::array<double, 3> x0;
      x0[0] = args.x0;
      x0[1] = args.y0;
      x0[2] = args.z0;

      Lorenz63_rk4_model model;
      model.time_step = args.time_step;
      model.state = x0;
      model.system.rho = args.rho;
      model.system.beta = args.beta;
      model.system.sigma = args.sigma;

      Eigen::VectorXd times(n_samples);
      Eigen::MatrixXd data(n_features, n_samples);

      generate_trajectory(model, n_samples, times, data, args.verbose);

      if (!args.trajectory_output_file.empty()) {
         if (args.verbose) {
            std::cout << "Writing trajectory to file: "
                      << args.trajectory_output_file << '\n';
         }
         write_trajectory_csv(
            args.trajectory_output_file, times, data);
      }

      const auto results = calculate_factorization(
         data, args.n_components, args.epsilon_states,
         args.n_folds, args.n_init, args.random_seed);

      if (!args.dictionary_output_file.empty()) {
         if (args.verbose) {
            std::cout << "Writing dictionary to file: "
                      << args.dictionary_output_file << '\n';
         }
         write_dictionary_csv(
            args.dictionary_output_file, results);
      }

      if (!args.affiliations_output_file.empty()) {
         if (args.verbose) {
            std::cout << "Writing affiliations to file: "
                      << args.affiliations_output_file << '\n';
         }
         write_affiliations_csv(
            args.affiliations_output_file, results);
      }

      if (!args.summary_output_file.empty()) {
         if (args.verbose) {
            std::cout << "Writing fit summary to file: "
                      << args.summary_output_file << '\n';
         }
         write_summary_csv(
            args.summary_output_file, results);
      } else {
         write_summary(results);
      }
   } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
   }

   return 0;
}
