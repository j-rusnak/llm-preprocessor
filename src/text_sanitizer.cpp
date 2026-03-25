#include "text_sanitizer.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace preprocessor {

std::string TextSanitizer::sanitize(const std::string& input) {
    // Lowercase the entire string.
    std::string result;
    result.reserve(input.size());
    for (unsigned char c : input) {
        result.push_back(static_cast<char>(std::tolower(c)));
    }

    // Collapse runs of whitespace into a single space and trim.
    std::istringstream stream(result);
    std::string word;
    std::string output;

    while (stream >> word) {
        if (!output.empty()) {
            output.push_back(' ');
        }
        output += word;
    }

    return output;
}

} // namespace preprocessor
