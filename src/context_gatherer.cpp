#include "context_gatherer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <curl/curl.h>

namespace preprocessor {

std::string ContextGatherer::read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::size_t ContextGatherer::write_callback(char* ptr, std::size_t size,
                                            std::size_t nmemb, std::string* data) {
    std::size_t total = size * nmemb;
    data->append(ptr, total);
    return total;
}

std::string ContextGatherer::fetch_url(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    std::string response_body;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "LLMPreprocessor/1.0");
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl, CURLOPT_MAXFILESIZE, 10L * 1024 * 1024); // 10 MB limit

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::string error = curl_easy_strerror(res);
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl request failed: " + error);
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code != 200) {
        throw std::runtime_error("HTTP request returned status " + std::to_string(http_code));
    }

    return response_body;
}

} // namespace preprocessor
