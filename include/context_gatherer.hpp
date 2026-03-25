#pragma once

#include <string>

namespace preprocessor {

class ContextGatherer {
public:
    static std::string read_file(const std::string& filepath);
    static std::string fetch_url(const std::string& url);

private:
    static std::size_t write_callback(char* ptr, std::size_t size,
                                      std::size_t nmemb, std::string* data);
};

} // namespace preprocessor
