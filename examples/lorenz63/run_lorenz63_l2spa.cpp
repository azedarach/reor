#include "lorenz63_rk4_model.hpp"

#include "reor/random_matrix.hpp"

#include "cross_validation.hpp"
#include "l2spa_fit_wrappers.hpp"

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
   int n_folds{1};
   int n_init{10};
   int n_samples{5000};
   int random_seed{0};
   double tolerance{1e-6};
   int max_iterations{1000000};
   double x0{1};
   double y0{1};
   double z0{1};
   double beta{8. / 3.};
   double sigma{10};
   double rho{28};
   double time_step{1e-2};
   std::string trajectory_output_file{""};
   std::string summary_output_file{""};
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
      "  -e, --epsilon-states=EPSILON_STATES regularization parameter\n"
      "  -f, --n-folds=N_FOLDS               number of cross-validation folds\n"
      "  -h, --help                          print this help message\n"
      "  -i, --n-init=N_INIT                 number of initializations\n"
      "  -k, --n-components=N_COMPONENTS     number of dictionary vectors\n"
      "  -m, --max-iterations=MAX_ITERATIONS maximum number of iterations\n"
      "  -o, --summary-output-file=FILE      file to write fit summaries to\n"
      "  --oos                               use out-of-sample cross-validation\n"
      "  -r, --random-seed=RANDOM_SEED       random seed\n"
      "  -s, --tolerance=TOLERANCE           stopping tolerance\n"
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

      if (opt == "-f") {
         if (i == argc) {
            throw std::runtime_error(
               "'-f' given but number of folds not provided");
         }
         const std::string n_folds(argv[i++]);
         if (starts_with(n_folds, "-")) {
            throw std::runtime_error(
               "'-f' given but valid number of folds not provided");
         }
         options.n_folds = std::stoi(n_folds);
         continue;
      }

      if (starts_with(opt, "--n-folds=")) {
         const std::string n_folds = get_option_value(opt);
         if (n_folds.empty() || starts_with(n_folds, "-")) {
            throw std::runtime_error(
               "'--n-folds=' given but valid number of folds not provided");
         }
         options.n_folds = std::stoi(n_folds);
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

      if (opt == "--oos") {
         options.out_of_sample_cv = true;
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

      throw std::runtime_error(
         "unrecognized command line option '" + opt + "'");
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

std::vector<Factorization_result> calculate_factorization(
   const Eigen::MatrixXd& data, const std::vector<int>& n_components,
   const std::vector<double>& epsilon_states,
   int n_folds, int n_init, double tolerance, int max_iterations,
   int random_seed, bool oos_cv, bool verbose)
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

   const int n_features = data.rows();
   const int n_samples = data.cols();

   Eigen::MatrixXd* initial_dictionary = nullptr;
   Eigen::MatrixXd* initial_affiliations = nullptr;

   std::size_t n_fits = 0;
   std::vector<Factorization_result> results;
   for (int k : n_components) {
      for (double eps : epsilon_states) {
         Eigen::MatrixXd dictionary_guess(
            Eigen::MatrixXd::Zero(n_features, k));
         Eigen::MatrixXd affiliations_guess(
            Eigen::MatrixXd::Zero(k, n_samples));

         if (n_fits > 0 && results[n_fits - 1].success) {
            dictionary_guess.block(0, 0, n_features, k - 1) =
               results[n_fits - 1].dictionary;
            affiliations_guess.block(0, 0, k - 1, n_samples) =
               results[n_fits - 1].affiliations;

            initial_dictionary = &dictionary_guess;
            initial_affiliations = &affiliations_guess;
         } else {
            initial_dictionary = nullptr;
            initial_affiliations = nullptr;
         }

         Factorization_result result = run_cross_validated_l2spa(
            data, test_sets, k, eps, n_init,
            tolerance, max_iterations, initial_dictionary,
            initial_affiliations, generator, verbose);

         results.push_back(result);
         ++n_fits;
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

void write_summary_csv(const std::string& output_file,
                       const std::vector<Factorization_result>& results)
{
   std::vector<std::string> fields;
   fields.push_back("n_components");
   fields.push_back("epsilon_states");
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
      data(2, i) = results[i].n_fits;
      data(3, i) = results[i].n_successful_fits;
      data(4, i) = results[i].n_successful_validations;
      data(5, i) = results[i].min_n_iter;
      data(6, i) = results[i].max_n_iter;
      data(7, i) = results[i].min_cost;
      data(8, i) = results[i].max_cost;
      data(9, i) = results[i].average_cost;
      data(10, i) = results[i].min_training_approx_rmse;
      data(11, i) = results[i].max_training_approx_rmse;
      data(12, i) = results[i].average_training_approx_rmse;
      data(13, i) = results[i].min_training_approx_rss;
      data(14, i) = results[i].max_training_approx_rss;
      data(15, i) = results[i].average_training_approx_rss;
      data(16, i) = results[i].min_test_approx_rmse;
      data(17, i) = results[i].max_test_approx_rmse;
      data(18, i) = results[i].average_test_approx_rmse;
      data(19, i) = results[i].min_test_approx_rss;
      data(20, i) = results[i].max_test_approx_rss;
      data(21, i) = results[i].average_test_approx_rss;
      data(22, i) = results[i].min_time_seconds;
      data(23, i) = results[i].max_time_seconds;
      data(24, i) = results[i].average_time_seconds;
      data(25, i) = results[i].success ? 1 : 0;
   }

   write_csv(output_file, data, fields);
}

void write_summary(const std::vector<Factorization_result>& results)
{
   std::vector<std::string> fields;
   fields.push_back("n_components");
   fields.push_back("epsilon_states");
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
      data(2, i) = results[i].n_fits;
      data(3, i) = results[i].n_successful_fits;
      data(4, i) = results[i].n_successful_validations;
      data(5, i) = results[i].min_n_iter;
      data(6, i) = results[i].max_n_iter;
      data(7, i) = results[i].min_cost;
      data(8, i) = results[i].max_cost;
      data(9, i) = results[i].average_cost;
      data(10, i) = results[i].min_training_approx_rmse;
      data(11, i) = results[i].max_training_approx_rmse;
      data(12, i) = results[i].average_training_approx_rmse;
      data(13, i) = results[i].min_training_approx_rss;
      data(14, i) = results[i].max_training_approx_rss;
      data(15, i) = results[i].average_training_approx_rss;
      data(16, i) = results[i].min_test_approx_rmse;
      data(17, i) = results[i].max_test_approx_rmse;
      data(18, i) = results[i].average_test_approx_rmse;
      data(19, i) = results[i].min_test_approx_rss;
      data(20, i) = results[i].max_test_approx_rss;
      data(21, i) = results[i].average_test_approx_rss;
      data(22, i) = results[i].min_time_seconds;
      data(23, i) = results[i].max_time_seconds;
      data(24, i) = results[i].average_time_seconds;
      data(25, i) = results[i].success ? 1 : 0;
   }

   write_header_line(std::cout, fields);
   write_data_lines(std::cout, data);
}

int main(int argc, const char* argv[])
{
   try {
      auto args = parse_cmd_line_args(argc, argv);

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

      if (args.n_components.empty()) {
         args.n_components.push_back(1);
      }

      if (args.epsilon_states.empty()) {
         args.epsilon_states.push_back(0);
      }

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

      if (args.n_folds < 1) {
         std::cerr << "Error: number of cross-validation folds must be"
                   << " at least one." << std::endl;
         return 1;
      }

      if (args.n_init < 1) {
         std::cerr << "Error: number of initialization must be at least one."
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
         args.n_folds, args.n_init, args.tolerance,
         args.max_iterations, args.random_seed,
         args.out_of_sample_cv, args.verbose);

      // if (!args.dictionary_output_file.empty()) {
      //    if (args.verbose) {
      //       std::cout << "Writing dictionary to file: "
      //                 << args.dictionary_output_file << '\n';
      //    }
      //    write_dictionary_csv(
      //       args.dictionary_output_file, results);
      // }

      // if (!args.affiliations_output_file.empty()) {
      //    if (args.verbose) {
      //       std::cout << "Writing affiliations to file: "
      //                 << args.affiliations_output_file << '\n';
      //    }
      //    write_affiliations_csv(
      //       args.affiliations_output_file, results);
      // }

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
