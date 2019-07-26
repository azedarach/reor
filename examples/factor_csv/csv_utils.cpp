#include "csv_utils.hpp"

#include <cstddef>

namespace reor {

bool starts_with(const std::string& line, const char* prefix)
{
   const std::string prefix_str(prefix);
   return starts_with(line, prefix_str);
}

bool starts_with(const std::string& line, const std::string& prefix)
{
   return !line.compare(0, prefix.size(), prefix);
}

namespace detail {

std::string ltrim(const std::string& s)
{
   if (s.find_first_not_of(" \t\n") == std::string::npos) {
      return "";
   } else {
      const auto pos = s.find_first_not_of(" \t\n");
      return s.substr(pos);
   }
}

std::string rtrim(const std::string& s)
{
   if (s.find_last_not_of(" \t\n") == std::string::npos) {
      return "";
   } else {
      const auto pos = s.find_last_not_of(" \t\n");
      return s.substr(0, pos + 1);
   }
}

std::string trim(const std::string& s)
{
   return ltrim(rtrim(s));
}

std::vector<std::string> split(
   const std::string& line, const char* delimiter)
{
   const std::string delimiter_str(delimiter);
   return split(line, delimiter_str);
}

std::vector<std::string> split(
   const std::string& line, const std::string& delimiter)
{
   std::vector<std::string> tokens;

   if (!line.empty() && line.find(delimiter) == std::string::npos) {
      tokens.push_back(line);
   } else {
      int pos = 0;
      while (line.find(delimiter, pos) != std::string::npos) {
         const auto delim_pos = line.find(delimiter, pos);
         tokens.push_back(line.substr(pos, delim_pos - pos));
         pos = delim_pos + 1;
      }
      tokens.push_back(line.substr(pos));
   }

   return tokens;
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

std::vector<std::string> parse_header_line(const std::string& line)
{
   std::string titles = trim(line);
   int comment_pos = titles.find('#');
   titles = trim(titles.substr(comment_pos + 1));

   std::vector<std::string> fields = split(titles, ",");
   const std::size_t n_fields = fields.size();
   for (std::size_t i = 0; i < n_fields; ++i) {
      fields[i] = trim(fields[i]);
   }

   return fields;
}

} // namespace detail

} // namespace reor
