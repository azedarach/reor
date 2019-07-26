#ifndef REOR_CSV_UTILS_HPP_INCLUDED
#define REOR_CSV_UTILS_HPP_INCLUDED

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace reor {

bool starts_with(const std::string&, const char*);
bool starts_with(const std::string&, const std::string&);

namespace detail {

std::string ltrim(const std::string&);
std::string rtrim(const std::string&);
std::string trim(const std::string&);
std::vector<std::string> split(const std::string&, const char*);
std::vector<std::string> split(const std::string&, const std::string&);
void write_header_line(std::ofstream&, const std::vector<std::string>&);
std::vector<std::string> parse_header_line(const std::string&);

template <class Matrix>
void write_data_lines(std::ofstream& ofs, const Matrix& data)
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

template <class Matrix>
void write_csv(const std::string& output_file, const Matrix& data,
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

template <class Matrix>
std::tuple<std::vector<std::string>, Matrix>
read_csv(const std::string& input_file)
{
   std::ifstream ifs(input_file);

   int n_features = -1;
   std::vector<std::string> fields;
   std::vector<std::vector<double> > data_lines;
   std::string line;

   bool has_header = false;

   while (std::getline(ifs, line)) {
      line = trim(line);

      if (line.empty()) {
         continue;
      }

      if (starts_with(line, "#") && !has_header) {
         fields = parse_header_line(line);
         continue;
      } else if (starts_with(line, "#")) {
         continue;
      }

      std::vector<std::string> fields = split(line, ",");
      const int n_fields = fields.size();

      if (n_features < 0) {
         n_features = n_fields;
      } else {
         if (n_fields != n_features) {
            throw std::runtime_error("incorrect number of fields");
         }
      }

      std::vector<double> values;
      for (int i = 0; i < n_fields; ++i) {
         values.push_back(std::stod(fields[i]));
      }

      data_lines.push_back(values);
   }

   const int n_samples = data_lines.size();

   Matrix data(n_features, n_samples);
   for (int i = 0; i < n_samples; ++i) {
      for (int j = 0; j < n_features; ++j) {
         data(j, i) = data_lines[i][j];
      }
   }

   return std::make_tuple(fields, data);
}

} // namespace detail

template <class Matrix>
struct CSV_writer {
   void write_csv(const std::string&, const Matrix&,
                  const std::vector<std::string>&) const;
};

template <class Matrix>
void CSV_writer<Matrix>::write_csv(
   const std::string& output_file, const Matrix& data,
   const std::vector<std::string>& fields) const
{
   detail::write_csv(output_file, data, fields);
}

template <class Matrix>
struct CSV_reader {
   std::tuple<std::vector<std::string>, Matrix> read_csv(
      const std::string&) const;
};

template <class Matrix>
std::tuple<std::vector<std::string>, Matrix>
CSV_reader<Matrix>::read_csv(const std::string& input_file) const
{
   return detail::read_csv<Matrix>(input_file);
}

} // namespace reor

#endif
