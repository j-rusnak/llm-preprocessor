#include "sync_endpoint.hpp"

#include "prompt_cache.hpp"

#include <nlohmann/json.hpp>

namespace preprocessor {

using nlohmann::json;

std::size_t SyncEndpoint::apply_to_cache(const SyncBundle& bundle, PromptCache* cache) {
    std::size_t n = 0;
    if (cache) {
        for (const auto& e : bundle.cache) {
            if (e.key.empty()) continue;
            cache->put(e.key, e.body);
            ++n;
        }
    }
    note_import();
    return n;
}

std::string SyncEndpoint::to_json(const SyncBundle& bundle) const {
    json j;
    j["cache"] = json::array();
    for (const auto& e : bundle.cache) {
        j["cache"].push_back({{"key", e.key}, {"body", e.body}});
    }
    j["vectors"] = json::array();
    for (const auto& v : bundle.vectors) {
        j["vectors"].push_back({
            {"chunk_id", v.chunk_id},
            {"vec", v.vec},
            {"source_path", v.source_path},
            {"text", v.text},
            {"start_line", v.start_line},
            {"end_line", v.end_line},
            {"symbol", v.symbol},
        });
    }
    return j.dump();
}

SyncBundle SyncEndpoint::from_json(const std::string& s) const {
    SyncBundle b;
    auto j = json::parse(s, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) return b;
    if (j.contains("cache") && j["cache"].is_array()) {
        for (const auto& e : j["cache"]) {
            SyncCacheEntry c;
            c.key = e.value("key", "");
            c.body = e.value("body", "");
            b.cache.push_back(std::move(c));
        }
    }
    if (j.contains("vectors") && j["vectors"].is_array()) {
        for (const auto& e : j["vectors"]) {
            SyncVectorEntry v;
            v.chunk_id = e.value("chunk_id", static_cast<std::uint64_t>(0));
            if (e.contains("vec") && e["vec"].is_array()) {
                for (const auto& f : e["vec"]) {
                    v.vec.push_back(f.get<float>());
                }
            }
            v.source_path = e.value("source_path", "");
            v.text = e.value("text", "");
            v.start_line = e.value("start_line", static_cast<std::size_t>(0));
            v.end_line = e.value("end_line", static_cast<std::size_t>(0));
            v.symbol = e.value("symbol", "");
            b.vectors.push_back(std::move(v));
        }
    }
    return b;
}

void SyncEndpoint::note_export() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    ++exports_;
}

void SyncEndpoint::note_import() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    ++imports_;
}

std::size_t SyncEndpoint::bundles_exported() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return exports_;
}

std::size_t SyncEndpoint::bundles_imported() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return imports_;
}

}  // namespace preprocessor
