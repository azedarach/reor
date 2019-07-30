#include "reor/random_matrix.hpp"

#include "cross_validation.hpp"
#include "csv_utils.hpp"
#include "l2trspa_fit_wrappers.hpp"

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
   std::vector<int> n_components{};
   std::vector<double> epsilon_states{};
   std::vector<double> epsilon_weights{};
   std::vector<double> eta_weights{};
   std::vector<double> epsilon_parameters{};
   std::vector<int> lag_set{};
   double evaluation_fraction{0.25};
   int n_init{10};
   int random_seed{0};
   double tolerance{1e-6};
   int max_iterations{1000000};
   std::string input_file{""};
   std::string summary_output_file{""};
   bool normalize{false};
   bool verbose{false};
};

void print_usage()
{
   std::cout <<
      "Usage: fit_csv_l2spa [OPTION] FILE\n\n"
      "Calculate l2-SPA factorizations for data given in CSV file.\n\n"
      "Example: fit_csv_l2spa -k 3 data.csv\n\n"
      "Options:\n"
      "  -e, --epsilon-states=EPSILON_STATES dictionary regularization parameter\n"
      "  -E, --epsilon-weights=EPSILON       weights regularization parameter\n"
      "  -f, --evaluation-fraction=FRACTION  validation fraction\n"
      "  -g, --eta-weights=ETA               weights norm regularization parameter\n"
      "  -h, --help                          print this help message\n"
      "  -i, --n-init=N_INIT                 number of initializations\n"
      "  -k, --n-components=N_COMPONENTS     number of dictionary vectors\n"
      "  -l, --lag=LAG                       lag to include in AR model\n"
      "  -m, --max-iterations=MAX_ITERATIONS maximum number of iterations\n"
      "  -n, --normalize                     normalize data to interval [-1, 1]\n"
      "  -o, --summary-output-file=FILE      file to write fit summaries to\n"
      "  -r, --random-seed=RANDOM_SEED       random seed\n"
      "  -s, --tolerance=TOLERANCE           stopping tolerance\n"
      "  -v, --verbose                       produce verbose output\n"
      "  -w, --epsilon-parameters=EPSILON    parameters regularization\n"
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

void parse_positional_args(
   const std::vector<std::string>& args, Program_options& options)
{
   if (args.size() != 1) {
      throw std::runtime_error("exactly one positional argument expected");
   }

   options.input_file = args[0];
}

Program_options parse_cmd_line_args(int argc, const char* argv[])
{
   Program_options options;

   std::vector<std::string> positional_args;
   int i = 1;
   while (i < argc) {
      const std::string opt(argv[i++]);

      if (opt == "-e") {
         if (i == argc) {
            throw std::runtime_error(
               "'-e' given but regularization parameter not provided");
         }
         const std::string epsilon_states(argv[i++]);
         if (starts_with(epsilon_states, "-")) {
            throw std::runtime_error(
               "'-e' given but valid regularization parameter not provided");
         }
         options.epsilon_states.push_back(std::stod(epsilon_states));
         continue;
      }

      if (starts_with(opt, "--epsilon-states=")) {
         const std::string epsilon_states = get_option_value(opt);
         if (epsilon_states.empty() || starts_with(epsilon_states, "-")) {
            throw std::runtime_error(
               "'--epsilon-states=' given but valid regularization parameter not provided");
         }
         options.epsilon_states.push_back(std::stod(epsilon_states));
         continue;
      }

      if (opt == "-E") {
         if (i == argc) {
            throw std::runtime_error(
               "'-E' given but regularization parameter not provided");
         }
         const std::string epsilon_weights(argv[i++]);
         if (starts_with(epsilon_weights, "-")) {
            throw std::runtime_error(
               "'-E' given but valid regularization parameter not provided");
         }
         options.epsilon_weights.push_back(std::stod(epsilon_weights));
         continue;
      }

      if (starts_with(opt, "--epsilon-weights=")) {
         const std::string epsilon_weights = get_option_value(opt);
         if (epsilon_weights.empty() ||
             starts_with(epsilon_weights, "-")) {
            throw std::runtime_error(
               "'--epsilon-weights=' given but valid regularization parameter not provided");
         }
         options.epsilon_weights.push_back(std::stod(epsilon_weights));
         continue;
      }

      if (opt == "-f") {
         if (i == argc) {
            throw std::runtime_error(
               "'-f' given but evaluation fraction not provided");
         }
         const std::string evaluation_fraction(argv[i++]);
         if (starts_with(evaluation_fraction, "-")) {
            throw std::runtime_error(
               "'-f' given but valid evaluation fraction not provided");
         }
         options.evaluation_fraction = std::stod(evaluation_fraction);
         continue;
      }

      if (starts_with(opt, "--evaluation-fraction=")) {
         const std::string evaluation_fraction = get_option_value(opt);
         if (evaluation_fraction.empty() ||
             starts_with(evaluation_fraction, "-")) {
            throw std::runtime_error(
               "'--evaluation-fraction=' given but valid evaluation fraction not provided");
         }
         options.evaluation_fraction = std::stod(evaluation_fraction);
         continue;
      }

      if (opt == "-g") {
         if (i == argc) {
            throw std::runtime_error(
               "'-g' given but regularization parameter not provided");
         }
         const std::string eta_weights(argv[i++]);
         if (starts_with(eta_weights, "-")) {
            throw std::runtime_error(
               "'-g' given but valid regularization parameter not provided");
         }
         options.eta_weights.push_back(std::stod(eta_weights));
         continue;
      }

      if (starts_with(opt, "--eta-weights=")) {
         const std::string eta_weights = get_option_value(opt);
         if (eta_weights.empty() || starts_with(eta_weights, "-")) {
            throw std::runtime_error(
               "'--eta-weights=' given but valid regularization parameter not provided");
         }
         options.eta_weights.push_back(std::stod(eta_weights));
         continue;
      }

      if (opt == "-h" || opt == "--help") {
         print_usage();
         exit(EXIT_SUCCESS);
      }

      if (opt == "-i") {
         if (i == argc) {
            throw std::runtime_error(
               "'-i' given but number of initializations not provided");
         }
         const std::string n_init(argv[i++]);
         if (starts_with(n_init, "-")) {
            throw std::runtime_error(
               "'-i' given but valid number of initializations not provided");
         }
         options.n_init = std::stoi(n_init);
         continue;
      }

      if (starts_with(opt, "--n-init=")) {
         const std::string n_init = get_option_value(opt);
         if (n_init.empty() || starts_with(n_init, "-")) {
            throw std::runtime_error(
               "'--n-init=' given but valid number of initializations not provided");
         }
         options.n_init = std::stoi(n_init);
         continue;
      }

      if (opt == "-k") {
         if (i == argc) {
            throw std::runtime_error(
               "'-k' given but number of components not provided");
         }
         const std::string n_components(argv[i++]);
         if (starts_with(n_components, "-")) {
            throw std::runtime_error(
               "'-k' given but valid number of components not provided");
         }
         options.n_components.push_back(std::stoi(n_components));
         continue;
      }

      if (starts_with(opt, "--n-components=")) {
         const std::string n_components = get_option_value(opt);
         if (n_components.empty() || starts_with(n_components, "-")) {
            throw std::runtime_error(
               "'--n-components=' given but valid number of components not provided");
         }
         options.n_components.push_back(std::stoi(n_components));
         continue;
      }

      if (opt == "-l") {
         if (i == argc) {
            throw std::runtime_error(
               "'-l' given but lag not provided");
         }
         const std::string lag(argv[i++]);
         if (starts_with(lag, "-")) {
            throw std::runtime_error(
               "'-l' given but valid lag not provided");
         }
         options.lag_set.push_back(std::stoi(lag));
         continue;
      }

      if (starts_with(opt, "--lag=")) {
         const std::string lag = get_option_value(opt);
         if (lag.empty() || starts_with(lag, "-")) {
            throw std::runtime_error(
               "'--lag=' given but valid lag not provided");
         }
         options.lag_set.push_back(std::stoi(lag));
         continue;
      }

      if (opt == "-m") {
         if (i == argc) {
            throw std::runtime_error(
               "'-m' given but maximum number of iterations not provided");
         }
         const std::string max_iterations(argv[i++]);
         if (starts_with(max_iterations, "-")) {
            throw std::runtime_error(
               "'-m' given but valid maximum number of iterations not provided");
         }
         options.max_iterations = std::stoi(max_iterations);
         continue;
      }

      if (starts_with(opt, "--max-iterations=")) {
         const std::string max_iterations = get_option_value(opt);
         if (max_iterations.empty() || starts_with(max_iterations, "-")) {
            throw std::runtime_error(
               "'--max-iterations=' given but valid maximum number of iterations not provided");
         }
         options.max_iterations = std::stoi(max_iterations);
         continue;
      }

      if (opt == "-n" || opt == "--normalize") {
         options.normalize = true;
         continue;
      }

      if (opt == "-o") {
         if (i == argc) {
            throw std::runtime_error(
               "'-o' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-o' given but no output file name provided");
         }
         options.summary_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--summary-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--summary-output-file=' given but no output file name provided");
         }
         options.summary_output_file = filename;
         continue;
      }

      if (opt == "-h" || opt == "--help") {
         print_usage();
         exit(EXIT_SUCCESS);
      }

      if (opt == "-r") {
         if (i == argc) {
            throw std::runtime_error(
               "'-r' given but random seed not provided");
         }
         const std::string random_seed(argv[i++]);
         if (starts_with(random_seed, "-")) {
            throw std::runtime_error(
               "'-r' given but valid random seed not provided");
         }
         options.random_seed = std::stoi(random_seed);
         continue;
      }

      if (starts_with(opt, "--random-seed=")) {
         const std::string random_seed = get_option_value(opt);
         if (random_seed.empty() || starts_with(random_seed, "-")) {
            throw std::runtime_error(
               "'--random-seed=' given but valid random seed not provided");
         }
         options.random_seed = std::stoi(random_seed);
         continue;
      }

      if (opt == "-s") {
         if (i == argc) {
            throw std::runtime_error(
               "'-s' given but tolerance not provided");
         }
         const std::string tolerance(argv[i++]);
         if (starts_with(tolerance, "-")) {
            throw std::runtime_error(
               "'-s' given but valid tolerance not provided");
         }
         options.tolerance = std::stod(tolerance);
         continue;
      }

      if (starts_with(opt, "--tolerance=")) {
         const std::string tolerance = get_option_value(opt);
         if (tolerance.empty() || starts_with(tolerance, "-")) {
            throw std::runtime_error(
               "'--tolerance=' given but valid tolerance not provided");
         }
         options.tolerance = std::stod(tolerance);
         continue;
      }

      if (opt == "-v" || opt == "--verbose") {
         options.verbose = true;
         continue;
      }

      if (opt == "-w") {
         if (i == argc) {
            throw std::runtime_error(
               "'-w' given but regularization parameter not provided");
         }
         const std::string epsilon_parameters(argv[i++]);
         if (starts_with(epsilon_parameters, "-")) {
            throw std::runtime_error(
               "'-w' given but valid regularization parameter not provided");
         }
         options.epsilon_parameters.push_back(
            std::stod(epsilon_parameters));
         continue;
      }

      if (starts_with(opt, "--epsilon-parameters=")) {
         const std::string epsilon_parameters = get_option_value(opt);
         if (epsilon_parameters.empty() ||
             starts_with(epsilon_parameters, "-")) {
            throw std::runtime_error(
               "'--epsilon-parameters=' given but valid regularization parameter not provided");
         }
         options.epsilon_parameters.push_back(std::stod(epsilon_parameters));
         continue;
      }

      positional_args.push_back(opt);
   }

   parse_positional_args(positional_args, options);

   return options;
}

int check_arguments(const Program_options& args)
{
   for (auto k : args.n_components) {
      if (k < 1) {
         std::cerr << "Error: number of components must be at least one."
                   << std::endl;
         return 1;
      }
   }

   for (auto eps: args.epsilon_states) {
      if (eps < 0) {
         std::cerr << "Error: regularization must be non-negative."
                   << std::endl;
         return 1;
      }
   }

   for (auto eps : args.epsilon_weights) {
      if (eps < 0) {
         std::cerr << "Error: regularization must be non-negative."
                   << std::endl;
         return 1;
      }
   }

   for (auto eta : args.eta_weights) {
      if (eta < 0) {
         std::cerr << "Error: regularization must be non-negative."
                   << std::endl;
         return 1;
      }
   }

   for (auto eps : args.epsilon_parameters) {
      if (eps < 0) {
         std::cerr << "Error: regularization must be non-negative."
                   << std::endl;
         return 1;
      }
   }

   for (auto l : args.lag_set) {
      if (l < 1) {
         std::cerr << "Error: lag parameter must be at least 1."
                   << std::endl;
         return 1;
      }
   }

   if (args.evaluation_fraction <= 0 ||
       args.evaluation_fraction >= 1) {
      std::cerr << "Error: evaluation fraction must be between 0 and 1."
                << std::endl;
      return 1;
   }

   if (args.n_init < 1) {
      std::cerr << "Error: number of initializations must be at least one."
                << std::endl;
      return 1;
   }

   if (args.max_iterations < 1) {
      std::cerr << "Error: maximum number of iterations must be at least one."
                << std::endl;
      return 1;
   }

   if (args.tolerance <= 0) {
      std::cerr << "Error: tolerance must be positive."
                << std::endl;
      return 1;
   }

   return 0;
}

void normalize_data_matrix(Eigen::MatrixXd& data)
{
   const auto n_features = data.rows();
   const auto n_samples = data.cols();

   const Eigen::VectorXd normalizations =
      data.cwiseAbs().rowwise().maxCoeff();

   for (int i = 0; i < n_features; ++i) {
      for (int j = 0; j < n_samples; ++j) {
         data(i, j) /= normalizations(i);
      }
   }
}

std::vector<Factorization_result> calculate_factorization(
   const Eigen::MatrixXd& data, const std::vector<int>& n_components,
   const std::vector<double>& epsilon_states,
   const std::vector<double>& epsilon_weights,
   const std::vector<double>& eta_weights,
   const std::vector<double>& epsilon_parameters,
   const std::vector<int>& lag_set,
   double evaluation_fraction,
   int n_init, double tolerance, int max_iterations,
   int random_seed, bool verbose)
{
   if (verbose) {
      std::cout << "Running factorization algorithm\n";
      std::cout << "Random seed: " << random_seed << '\n';
   }

   const auto start_time = std::chrono::high_resolution_clock::now();

   std::mt19937 generator(random_seed);

   const int n_features = data.rows();
   const int n_samples = data.cols();

   Eigen::MatrixXd* initial_dictionary = nullptr;
   Eigen::MatrixXd* initial_weights = nullptr;

   std::size_t n_fits = 0;
   std::vector<Factorization_result> results;
   for (int k : n_components) {
      for (double eps_S : epsilon_states) {
         for (double eps_Gamma : epsilon_weights) {
            for (double eta_Gamma : eta_weights) {
               for (double eps_W : epsilon_parameters) {

                  Eigen::MatrixXd dictionary_guess(
                     Eigen::MatrixXd::Zero(n_features, k));
                  Eigen::MatrixXd weights_guess(
                     Eigen::MatrixXd::Zero(k, n_samples));

                  if (n_fits > 0 && results[n_fits - 1].success) {
                     const auto previous_n_components =
                        backends::rows(results[n_fits - 1].weights);
                     dictionary_guess.block(
                        0, 0, n_features, previous_n_components - 1) =
                        results[n_fits - 1].dictionary;
                     weights_guess.block(
                        0, 0, previous_n_components - 1, n_samples) =
                        results[n_fits - 1].weights;

                     initial_dictionary = &dictionary_guess;
                     initial_weights = &weights_guess;
                  } else {
                     initial_dictionary = nullptr;
                     initial_weights = nullptr;
                  }

                  Factorization_result result = run_and_evaluate_l2trspa(
                     data, evaluation_fraction,
                     k, eps_S, eps_Gamma, eta_Gamma, eps_W,
                     lag_set, n_init,
                     tolerance, max_iterations, initial_dictionary,
                     initial_weights, generator, verbose);

                  results.push_back(result);
                  ++n_fits;
               }
            }
         }
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
   std::ostream& ofs, const std::vector<std::string>& fields)
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

void write_data_lines(std::ostream& ofs, const Eigen::MatrixXd& data)
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

void write_summary_csv(const std::string& output_file,
                       const std::vector<Factorization_result>& results)
{
   std::vector<std::string> fields;
   fields.push_back("n_components");
   fields.push_back("epsilon_states");
   fields.push_back("epsilon_weights");
   fields.push_back("eta_weights");
   fields.push_back("epsilon_parameters");
   fields.push_back("n_fits");
   fields.push_back("n_successful_fits");
   fields.push_back("n_successful_validations");
   fields.push_back("min_n_iter");
   fields.push_back("max_n_iter");
   fields.push_back("min_cost");
   fields.push_back("max_cost");
   fields.push_back("average_cost");
   fields.push_back("min_training_approx_rmse");
   fields.push_back("max_training_approx_rmse");
   fields.push_back("average_training_approx_rmse");
   fields.push_back("min_training_approx_rss");
   fields.push_back("max_training_approx_rss");
   fields.push_back("average_training_approx_rss");
   fields.push_back("min_test_approx_rmse");
   fields.push_back("max_test_approx_rmse");
   fields.push_back("average_test_approx_rmse");
   fields.push_back("min_test_approx_rss");
   fields.push_back("max_test_approx_rss");
   fields.push_back("average_test_approx_rss");
   fields.push_back("min_test_pred_rmse");
   fields.push_back("max_test_pred_rmse");
   fields.push_back("average_test_pred_rmse");
   fields.push_back("min_test_pred_rss");
   fields.push_back("max_test_pred_rss");
   fields.push_back("average_test_pred_rss");
   fields.push_back("min_time_seconds");
   fields.push_back("max_time_seconds");
   fields.push_back("average_time_seconds");
   fields.push_back("success");

   std::size_t n_factorizations = results.size();
   std::size_t n_fields = fields.size();
   Eigen::MatrixXd data(n_fields, n_factorizations);
   for (std::size_t i = 0; i < n_factorizations; ++i) {
      data(0, i) = results[i].n_components;
      data(1, i) = results[i].epsilon_states;
      data(2, i) = results[i].epsilon_weights;
      data(3, i) = results[i].eta_weights;
      data(4, i) = results[i].epsilon_parameters;
      data(5, i) = results[i].n_fits;
      data(6, i) = results[i].n_successful_fits;
      data(7, i) = results[i].n_successful_validations;
      data(8, i) = results[i].min_n_iter;
      data(9, i) = results[i].max_n_iter;
      data(10, i) = results[i].min_cost;
      data(11, i) = results[i].max_cost;
      data(12, i) = results[i].average_cost;
      data(13, i) = results[i].min_training_approx_rmse;
      data(14, i) = results[i].max_training_approx_rmse;
      data(15, i) = results[i].average_training_approx_rmse;
      data(16, i) = results[i].min_training_approx_rss;
      data(17, i) = results[i].max_training_approx_rss;
      data(18, i) = results[i].average_training_approx_rss;
      data(19, i) = results[i].min_test_approx_rmse;
      data(20, i) = results[i].max_test_approx_rmse;
      data(21, i) = results[i].average_test_approx_rmse;
      data(22, i) = results[i].min_test_approx_rss;
      data(23, i) = results[i].max_test_approx_rss;
      data(24, i) = results[i].average_test_approx_rss;
      data(25, i) = results[i].min_test_pred_rmse;
      data(26, i) = results[i].max_test_pred_rmse;
      data(27, i) = results[i].average_test_pred_rmse;
      data(28, i) = results[i].min_test_pred_rss;
      data(29, i) = results[i].max_test_pred_rss;
      data(30, i) = results[i].average_test_pred_rss;
      data(31, i) = results[i].min_time_seconds;
      data(32, i) = results[i].max_time_seconds;
      data(33, i) = results[i].average_time_seconds;
      data(34, i) = results[i].success ? 1 : 0;
   }

   write_csv(output_file, data, fields);
}

void write_summary(const std::vector<Factorization_result>& results)
{
   std::vector<std::string> fields;
   fields.push_back("n_components");
   fields.push_back("epsilon_states");
   fields.push_back("epsilon_weights");
   fields.push_back("eta_weights");
   fields.push_back("epsilon_parameters");
   fields.push_back("n_fits");
   fields.push_back("n_successful_fits");
   fields.push_back("n_successful_validations");
   fields.push_back("min_n_iter");
   fields.push_back("max_n_iter");
   fields.push_back("min_cost");
   fields.push_back("max_cost");
   fields.push_back("average_cost");
   fields.push_back("min_training_approx_rmse");
   fields.push_back("max_training_approx_rmse");
   fields.push_back("average_training_approx_rmse");
   fields.push_back("min_training_approx_rss");
   fields.push_back("max_training_approx_rss");
   fields.push_back("average_training_approx_rss");
   fields.push_back("min_test_approx_rmse");
   fields.push_back("max_test_approx_rmse");
   fields.push_back("average_test_approx_rmse");
   fields.push_back("min_test_approx_rss");
   fields.push_back("max_test_approx_rss");
   fields.push_back("average_test_approx_rss");
   fields.push_back("min_test_pred_rmse");
   fields.push_back("max_test_pred_rmse");
   fields.push_back("average_test_pred_rmse");
   fields.push_back("min_test_pred_rss");
   fields.push_back("max_test_pred_rss");
   fields.push_back("average_test_pred_rss");
   fields.push_back("min_time_seconds");
   fields.push_back("max_time_seconds");
   fields.push_back("average_time_seconds");
   fields.push_back("success");

   std::size_t n_factorizations = results.size();
   std::size_t n_fields = fields.size();
   Eigen::MatrixXd data(n_fields, n_factorizations);
   for (std::size_t i = 0; i < n_factorizations; ++i) {
            data(0, i) = results[i].n_components;
      data(1, i) = results[i].epsilon_states;
      data(2, i) = results[i].epsilon_weights;
      data(3, i) = results[i].eta_weights;
      data(4, i) = results[i].epsilon_parameters;
      data(5, i) = results[i].n_fits;
      data(6, i) = results[i].n_successful_fits;
      data(7, i) = results[i].n_successful_validations;
      data(8, i) = results[i].min_n_iter;
      data(9, i) = results[i].max_n_iter;
      data(10, i) = results[i].min_cost;
      data(11, i) = results[i].max_cost;
      data(12, i) = results[i].average_cost;
      data(13, i) = results[i].min_training_approx_rmse;
      data(14, i) = results[i].max_training_approx_rmse;
      data(15, i) = results[i].average_training_approx_rmse;
      data(16, i) = results[i].min_training_approx_rss;
      data(17, i) = results[i].max_training_approx_rss;
      data(18, i) = results[i].average_training_approx_rss;
      data(19, i) = results[i].min_test_approx_rmse;
      data(20, i) = results[i].max_test_approx_rmse;
      data(21, i) = results[i].average_test_approx_rmse;
      data(22, i) = results[i].min_test_approx_rss;
      data(23, i) = results[i].max_test_approx_rss;
      data(24, i) = results[i].average_test_approx_rss;
      data(25, i) = results[i].min_test_pred_rmse;
      data(26, i) = results[i].max_test_pred_rmse;
      data(27, i) = results[i].average_test_pred_rmse;
      data(28, i) = results[i].min_test_pred_rss;
      data(29, i) = results[i].max_test_pred_rss;
      data(30, i) = results[i].average_test_pred_rss;
      data(31, i) = results[i].min_time_seconds;
      data(32, i) = results[i].max_time_seconds;
      data(33, i) = results[i].average_time_seconds;
      data(34, i) = results[i].success ? 1 : 0;
   }

   write_header_line(std::cout, fields);
   write_data_lines(std::cout, data);
}

int main(int argc, const char* argv[])
{
   try {
      auto args = parse_cmd_line_args(argc, argv);

      if (args.n_components.empty()) {
         args.n_components.push_back(1);
      }

      if (args.epsilon_states.empty()) {
         args.epsilon_states.push_back(0);
      }

      if (args.epsilon_weights.empty()) {
         args.epsilon_weights.push_back(0);
      }

      if (args.eta_weights.empty()) {
         args.eta_weights.push_back(0);
      }

      if (args.epsilon_parameters.empty()) {
         args.epsilon_parameters.push_back(0);
      }

      if (args.lag_set.empty()) {
         args.lag_set.push_back(1);
      }

      const auto invalid_args = check_arguments(args);
      if (invalid_args) {
         return invalid_args;
      }

      if (args.verbose) {
         std::cout << "Reading data from file: " << args.input_file << '\n';
      }

      CSV_reader<Eigen::MatrixXd> csv_reader;
      const auto input_data = csv_reader.read_csv(args.input_file);

      const Eigen::MatrixXd initial_data = std::get<1>(input_data);

      Eigen::MatrixXd data(initial_data);
      if (args.normalize) {
         normalize_data_matrix(data);
      }

      if (args.verbose) {
         const int n_features = data.rows();
         const int n_samples = data.cols();
         std::cout << "Number of features: " << n_features << '\n';
         std::cout << "Number of samples: " << n_samples << '\n';
      }

      const auto results = calculate_factorization(
         data, args.n_components, args.epsilon_states,
         args.epsilon_weights, args.eta_weights,
         args.epsilon_parameters, args.lag_set,
         args.evaluation_fraction, args.n_init, args.tolerance,
         args.max_iterations, args.random_seed,
         args.verbose);

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
