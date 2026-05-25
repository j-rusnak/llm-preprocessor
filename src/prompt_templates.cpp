#include "prompt_templates.hpp"

#include <fstream>
#include <stdexcept>

#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

namespace preprocessor {

namespace {

constexpr const char* kDefaultCommonHeader =
    "{% if length(project_card) > 0 %}{{ project_card }}\n\n{% endif %}"
    "{% if length(context) > 0 %}Retrieved code context (most relevant first):\n{{ context }}\n\n{% endif %}";

constexpr const char* kDefaultCodeEdit =
    "You are an expert C++ engineer making a focused code edit.\n"
    "Use ONLY the retrieved code context below as ground truth; do not invent "
    "symbols, file paths, or APIs. When you propose a change, show a minimal "
    "diff or the exact replacement snippet plus the file path and line range.\n\n";

constexpr const char* kDefaultCodeExplain =
    "You are an expert code reviewer explaining existing code.\n"
    "Ground every claim in the retrieved code context below. If the context "
    "is insufficient, say so explicitly instead of guessing.\n\n";

constexpr const char* kDefaultCodeGenerate =
    "You are an expert C++ engineer writing new code.\n"
    "Match the conventions visible in the retrieved code context (naming, "
    "namespaces, error handling). Prefer the patterns already in use over "
    "novel idioms.\n\n";

constexpr const char* kDefaultMetaQuery =
    "You are answering a structural question about a software project.\n"
    "Use the project card and retrieved code context below as the source of "
    "truth. Cite specific file paths in your answer.\n\n";

constexpr const char* kDefaultFreeform =
    "You are a helpful AI coding assistant.\n"
    "Prefer the retrieved code context below over your training data when "
    "they conflict.\n\n";

std::string default_for(PromptBucket bucket) {
    std::string body;
    switch (bucket) {
        case PromptBucket::CodeEdit:     body = kDefaultCodeEdit;     break;
        case PromptBucket::CodeExplain:  body = kDefaultCodeExplain;  break;
        case PromptBucket::CodeGenerate: body = kDefaultCodeGenerate; break;
        case PromptBucket::MetaQuery:    body = kDefaultMetaQuery;    break;
        case PromptBucket::Freeform:     body = kDefaultFreeform;     break;
    }
    body += kDefaultCommonHeader;
    return body;
}

inja::Environment make_env() {
    inja::Environment env;
    env.set_trim_blocks(false);
    env.set_lstrip_blocks(false);
    return env;
}

} // namespace

PromptTemplates::PromptTemplates() {
    for (auto b : {PromptBucket::CodeEdit, PromptBucket::CodeExplain,
                   PromptBucket::CodeGenerate, PromptBucket::MetaQuery,
                   PromptBucket::Freeform}) {
        templates_[to_string(b)] = default_for(b);
    }
}

PromptTemplates PromptTemplates::load_from_file(const std::string& path) {
    PromptTemplates t;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("PromptTemplates: cannot open " + path);
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("PromptTemplates: invalid JSON: ") + e.what());
    }
    if (!j.is_object()) {
        throw std::runtime_error("PromptTemplates: top-level JSON must be an object");
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!it.value().is_string()) {
            throw std::runtime_error("PromptTemplates: value for '" + it.key() + "' must be a string");
        }
        // Validate the bucket name strictly: unknown keys would silently be
        // ignored which is a footgun.
        const std::string key = it.key();
        const PromptBucket b = bucket_from_string(key);
        if (to_string(b) != key) {
            throw std::runtime_error("PromptTemplates: unknown bucket '" + key + "'");
        }
        t.templates_[key] = it.value().get<std::string>();
    }
    return t;
}

void PromptTemplates::set_template(PromptBucket bucket, std::string source) {
    templates_[to_string(bucket)] = std::move(source);
}

const std::string& PromptTemplates::source_for(PromptBucket bucket) const {
    auto it = templates_.find(to_string(bucket));
    if (it == templates_.end()) {
        // Should be unreachable given the constructor populates every bucket,
        // but stay safe.
        static const std::string empty;
        return empty;
    }
    return it->second;
}

std::string PromptTemplates::render(PromptBucket bucket,
                                    const PromptRenderInput& input) const {
    nlohmann::json data;
    data["project_card"] = input.project_card;
    data["context"]      = input.context;
    data["user_message"] = input.user_message;
    data["bucket"]       = input.bucket.empty() ? to_string(bucket) : input.bucket;

    inja::Environment env = make_env();
    try {
        return env.render(source_for(bucket), data);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("PromptTemplates::render: ") + e.what());
    }
}

} // namespace preprocessor
