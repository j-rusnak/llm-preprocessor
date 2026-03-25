#pragma once

#include <string>

namespace preprocessor {

class TextSanitizer {
public:
    static std::string sanitize(const std::string& input);
};

} // namespace preprocessor
