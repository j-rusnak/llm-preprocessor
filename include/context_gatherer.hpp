#pragma once

#include <string>
#include <vector>

namespace preprocessor {

class ContextGatherer {
public:
    static std::string fetch_url(const std::string& url);
    static std::vector<std::string> extract_urls(const std::string& text);

private:
    static std::size_t write_callback(char* ptr, std::size_t size,
                                      std::size_t nmemb, std::string* data);
};

} // namespace preprocessor
