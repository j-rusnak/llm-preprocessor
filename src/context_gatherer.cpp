#include "context_gatherer.hpp"

#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

#include <curl/curl.h>

namespace preprocessor {

std::size_t ContextGatherer::write_callback(char* ptr, std::size_t size,
                                            std::size_t nmemb, std::string* data) {
    static constexpr std::size_t max_response_size = 10UL * 1024 * 1024; // 10 MB
    std::size_t total = size * nmemb;
    if (data->size() + total > max_response_size) {
        return 0; // Abort transfer — signals error to libcurl.
    }
    data->append(ptr, total);
    return total;
}

std::string ContextGatherer::fetch_url(const std::string& url) {
    auto curl = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(
        curl_easy_init(), curl_easy_cleanup);
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }

    std::string response_body;

    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "LLMPreprocessor/1.0");
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl.get(), CURLOPT_MAXFILESIZE, 10L * 1024 * 1024); // 10 MB limit

    CURLcode res = curl_easy_perform(curl.get());
    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl request failed: ") + curl_easy_strerror(res));
    }

    long http_code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code != 200) {
        throw std::runtime_error("HTTP request returned status " + std::to_string(http_code));
    }

    return response_body;
}

std::vector<std::string> ContextGatherer::extract_urls(const std::string& text) {
    std::vector<std::string> urls;
    const std::string http_prefix = "http://";
    const std::string https_prefix = "https://";

    // Characters that are valid in a URL (RFC 3986 unreserved + common reserved).
    auto is_url_char = [](char c) -> bool {
        if (std::isalnum(static_cast<unsigned char>(c))) return true;
        // unreserved: - . _ ~
        // reserved subset commonly in URLs: : / ? # [ ] @ ! $ & ' ( ) * + , ; = %
        static const std::string allowed = "-._~:/?#[]@!$&'()*+,;=%";
        return allowed.find(c) != std::string::npos;
    };

    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t start = std::string::npos;

        std::size_t https_pos = text.find(https_prefix, pos);
        std::size_t http_pos = text.find(http_prefix, pos);

        // Pick whichever comes first.
        if (https_pos != std::string::npos && (http_pos == std::string::npos || https_pos <= http_pos)) {
            start = https_pos;
        } else if (http_pos != std::string::npos) {
            start = http_pos;
        }

        if (start == std::string::npos) break;

        // Walk forward collecting URL characters.
        std::size_t end = start;
        while (end < text.size() && is_url_char(text[end])) {
            ++end;
        }

        // Strip trailing punctuation that's likely sentence-ending, not part of the URL.
        while (end > start && (text[end - 1] == '.' || text[end - 1] == ',' ||
               text[end - 1] == ')' || text[end - 1] == ';')) {
            --end;
        }

        std::string url = text.substr(start, end - start);
        if (url.size() > 10) { // Must be longer than just "http://x"
            urls.push_back(std::move(url));
        }

        pos = end;
    }

    return urls;
}

} // namespace preprocessor
