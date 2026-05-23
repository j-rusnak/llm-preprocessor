#include "symbol_graph.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_set>

namespace preprocessor {

std::string to_string(SymbolKind kind) {
    switch (kind) {
        case SymbolKind::Function: return "function";
        case SymbolKind::Method:   return "method";
        case SymbolKind::Class:    return "class";
        case SymbolKind::Struct:   return "struct";
        case SymbolKind::Enum:     return "enum";
        case SymbolKind::Variable: return "variable";
        case SymbolKind::Macro:    return "macro";
        case SymbolKind::Unknown:
        default:                   return "unknown";
    }
}

// ---------------------------------------------------------------------------
// RegexSymbolExtractor
// ---------------------------------------------------------------------------

namespace {

// Words that look like identifiers but are control keywords / built-ins we
// should NOT treat as call targets. Keeping this conservative — false
// positives only hurt graph density, not correctness.
const std::unordered_set<std::string>& reserved_words() {
    static const std::unordered_set<std::string> kSet = {
        "if", "else", "for", "while", "do", "switch", "case", "default",
        "break", "continue", "return", "goto", "throw", "try", "catch",
        "new", "delete", "this", "true", "false", "null", "nullptr",
        "and", "or", "not", "is", "in", "lambda", "def", "class", "struct",
        "enum", "union", "typedef", "using", "namespace", "template",
        "typename", "auto", "const", "constexpr", "static", "extern",
        "inline", "virtual", "override", "final", "public", "private",
        "protected", "void", "int", "long", "short", "char", "bool",
        "float", "double", "unsigned", "signed", "size_t", "uint8_t",
        "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
        "int64_t", "string", "std", "operator", "sizeof", "typeof",
        "var", "let", "const", "function", "fn", "pub", "mod", "use",
        "self", "super", "as", "import", "from", "package", "func",
        "type", "interface", "implements", "extends", "abstract", "yield",
        "await", "async"
    };
    return kSet;
}

// Per-line tracking of 1-based line numbers within chunk text. The chunk's
// absolute start_line maps line index 1 -> chunk.start_line.
std::size_t absolute_line(const CodeChunk& chunk, std::size_t local_line) {
    return chunk.start_line + (local_line > 0 ? local_line - 1 : 0);
}

bool is_ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Walk source counting lines so we can map a byte offset to a 1-based
// line index.
std::size_t line_at(const std::string& src, std::size_t pos) {
    std::size_t line = 1;
    for (std::size_t i = 0; i < pos && i < src.size(); ++i) {
        if (src[i] == '\n') ++line;
    }
    return line;
}

void emit_def(ExtractedSymbols& out, const CodeChunk& chunk,
              const std::string& name, SymbolKind kind, std::size_t local_line) {
    if (name.empty() || reserved_words().count(name)) return;
    SymbolDef d;
    d.name = name;
    d.kind = kind;
    d.file_path = chunk.file_path;
    d.line = absolute_line(chunk, local_line);
    d.chunk_id = chunk.id;
    out.defs.push_back(std::move(d));
}

void emit_ref(ExtractedSymbols& out, const CodeChunk& chunk,
              const std::string& name, std::size_t local_line) {
    if (name.empty() || reserved_words().count(name)) return;
    SymbolRef r;
    r.name = name;
    r.file_path = chunk.file_path;
    r.line = absolute_line(chunk, local_line);
    r.chunk_id = chunk.id;
    out.refs.push_back(std::move(r));
}

// Strip simple // line comments and /* */ block comments and quoted strings
// from a copy of the source so we don't extract identifiers from comments /
// string literals. Keeps newlines so line numbers are preserved.
std::string strip_comments_and_strings(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    enum class S { Code, LineComment, BlockComment, DQuote, SQuote, BTick };
    S state = S::Code;
    for (std::size_t i = 0; i < src.size(); ++i) {
        char c = src[i];
        char next = (i + 1 < src.size()) ? src[i + 1] : '\0';
        switch (state) {
            case S::Code:
                if (c == '/' && next == '/') { state = S::LineComment; out += "  "; ++i; }
                else if (c == '/' && next == '*') { state = S::BlockComment; out += "  "; ++i; }
                else if (c == '"') { state = S::DQuote; out += ' '; }
                else if (c == '\'') { state = S::SQuote; out += ' '; }
                else if (c == '`') { state = S::BTick; out += ' '; }
                else out += c;
                break;
            case S::LineComment:
                if (c == '\n') { state = S::Code; out += '\n'; }
                else out += ' ';
                break;
            case S::BlockComment:
                if (c == '*' && next == '/') { state = S::Code; out += "  "; ++i; }
                else out += (c == '\n' ? '\n' : ' ');
                break;
            case S::DQuote:
                if (c == '\\' && next != '\0') { out += "  "; ++i; }
                else if (c == '"') { state = S::Code; out += ' '; }
                else out += (c == '\n' ? '\n' : ' ');
                break;
            case S::SQuote:
                if (c == '\\' && next != '\0') { out += "  "; ++i; }
                else if (c == '\'') { state = S::Code; out += ' '; }
                else out += (c == '\n' ? '\n' : ' ');
                break;
            case S::BTick:
                if (c == '\\' && next != '\0') { out += "  "; ++i; }
                else if (c == '`') { state = S::Code; out += ' '; }
                else out += (c == '\n' ? '\n' : ' ');
                break;
        }
    }
    return out;
}

} // namespace

ExtractedSymbols RegexSymbolExtractor::extract(const CodeChunk& chunk) const {
    ExtractedSymbols out;
    const std::string cleaned = strip_comments_and_strings(chunk.text);

    // --- Definitions ---------------------------------------------------------
    // C/C++/Java/JS function-ish:  Type name(args) {
    //   Capture: identifier immediately before "(" preceded by a type-ish
    //   token at top-of-statement context. Match anywhere; this is a rough
    //   heuristic.
    static const std::regex kFunc(
        R"(\b([A-Za-z_][A-Za-z0-9_:]*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?\{)");
    // class/struct/enum NAME
    static const std::regex kClass(
        R"(\b(class|struct|enum(?:\s+class)?)\s+([A-Za-z_][A-Za-z0-9_]*))");
    // python def / class
    static const std::regex kPyDef(R"(\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\()");
    static const std::regex kPyCls(R"(\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:\(])");
    // rust fn NAME / go func NAME
    static const std::regex kRustFn(R"(\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<\(])");
    static const std::regex kGoFunc(R"(\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\()");
    // C macro
    static const std::regex kMacro(R"(^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*))");

    auto scan = [&](const std::regex& re, SymbolKind kind, int name_group) {
        auto begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const auto& m = *it;
            std::size_t pos = static_cast<std::size_t>(m.position(0));
            std::size_t line = line_at(cleaned, pos);
            emit_def(out, chunk, m.str(name_group), kind, line);
        }
    };

    scan(kFunc, SymbolKind::Function, 1);
    scan(kPyDef, SymbolKind::Function, 1);
    scan(kPyCls, SymbolKind::Class, 1);
    scan(kRustFn, SymbolKind::Function, 1);
    scan(kGoFunc, SymbolKind::Function, 1);
    scan(kMacro, SymbolKind::Macro, 1);

    {
        auto begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), kClass);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const auto& m = *it;
            std::size_t pos = static_cast<std::size_t>(m.position(0));
            std::size_t line = line_at(cleaned, pos);
            const std::string kw = m.str(1);
            SymbolKind k = SymbolKind::Class;
            if (kw.rfind("struct", 0) == 0) k = SymbolKind::Struct;
            else if (kw.rfind("enum", 0) == 0) k = SymbolKind::Enum;
            emit_def(out, chunk, m.str(2), k, line);
        }
    }

    // --- References ----------------------------------------------------------
    // identifier immediately followed by '(' (call site). Skip if the
    // identifier was just defined on the same line (avoids self-reference).
    static const std::regex kCall(R"(([A-Za-z_][A-Za-z0-9_]{1,63})\s*\()");
    std::unordered_set<std::string> def_names_this_chunk;
    for (const auto& d : out.defs) def_names_this_chunk.insert(d.name);

    auto begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), kCall);
    auto end = std::sregex_iterator();
    std::unordered_set<std::string> already_emitted_this_chunk;
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        std::string name = m.str(1);
        if (def_names_this_chunk.count(name)) continue; // own definition
        if (already_emitted_this_chunk.count(name)) continue; // dedupe per chunk
        std::size_t pos = static_cast<std::size_t>(m.position(0));
        // Filter: preceding non-space char must NOT be an identifier char
        // (avoids matching middle of `foo_bar(` after we already considered
        // foo_bar). Regex word boundary handles this in most cases, but
        // identifiers starting with '_' don't always behave that way.
        if (pos > 0) {
            char prev = cleaned[pos - 1];
            if (is_ident_char(prev)) continue;
        }
        already_emitted_this_chunk.insert(name);
        emit_ref(out, chunk, name, line_at(cleaned, pos));
    }

    return out;
}

// ---------------------------------------------------------------------------
// SymbolGraph
// ---------------------------------------------------------------------------

SymbolGraph::SymbolGraph() = default;
SymbolGraph::~SymbolGraph() = default;

namespace {

template <typename T>
void erase_chunk_from_name_index(
    std::unordered_map<std::string, std::vector<T>>& by_name,
    std::uint64_t chunk_id) {
    for (auto it = by_name.begin(); it != by_name.end();) {
        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [&](const T& v) { return v.chunk_id == chunk_id; }),
                  vec.end());
        if (vec.empty()) it = by_name.erase(it);
        else ++it;
    }
}

} // namespace

void SymbolGraph::update_chunk(const CodeChunk& chunk,
                               const ExtractedSymbols& extracted) {
    std::lock_guard<std::mutex> lock(mu_);

    // Drop prior contributions.
    auto def_it = defs_by_chunk_.find(chunk.id);
    if (def_it != defs_by_chunk_.end()) {
        def_total_ -= def_it->second.size();
        defs_by_chunk_.erase(def_it);
    }
    auto ref_it = refs_by_chunk_.find(chunk.id);
    if (ref_it != refs_by_chunk_.end()) {
        ref_total_ -= ref_it->second.size();
        refs_by_chunk_.erase(ref_it);
    }
    erase_chunk_from_name_index(defs_by_name_, chunk.id);
    erase_chunk_from_name_index(refs_by_name_, chunk.id);

    // Insert fresh.
    for (const auto& d : extracted.defs) {
        defs_by_name_[d.name].push_back(d);
        defs_by_chunk_[chunk.id].push_back(d);
    }
    for (const auto& r : extracted.refs) {
        refs_by_name_[r.name].push_back(r);
        refs_by_chunk_[chunk.id].push_back(r);
    }
    def_total_ += extracted.defs.size();
    ref_total_ += extracted.refs.size();

    chunks_by_file_[chunk.file_path].insert(chunk.id);
}

void SymbolGraph::remove_chunk(std::uint64_t chunk_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto def_it = defs_by_chunk_.find(chunk_id);
    if (def_it != defs_by_chunk_.end()) {
        def_total_ -= def_it->second.size();
        defs_by_chunk_.erase(def_it);
    }
    auto ref_it = refs_by_chunk_.find(chunk_id);
    if (ref_it != refs_by_chunk_.end()) {
        ref_total_ -= ref_it->second.size();
        refs_by_chunk_.erase(ref_it);
    }
    erase_chunk_from_name_index(defs_by_name_, chunk_id);
    erase_chunk_from_name_index(refs_by_name_, chunk_id);
    for (auto it = chunks_by_file_.begin(); it != chunks_by_file_.end();) {
        it->second.erase(chunk_id);
        if (it->second.empty()) it = chunks_by_file_.erase(it);
        else ++it;
    }
}

void SymbolGraph::remove_file(const std::string& file_path) {
    std::vector<std::uint64_t> ids;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = chunks_by_file_.find(file_path);
        if (it == chunks_by_file_.end()) return;
        ids.assign(it->second.begin(), it->second.end());
    }
    for (auto id : ids) remove_chunk(id);
}

std::vector<SymbolDef>
SymbolGraph::find_definitions(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = defs_by_name_.find(name);
    if (it == defs_by_name_.end()) return {};
    return it->second;
}

std::vector<SymbolRef>
SymbolGraph::find_references(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = refs_by_name_.find(name);
    if (it == refs_by_name_.end()) return {};
    return it->second;
}

std::vector<std::uint64_t>
SymbolGraph::neighbors_of(const std::vector<std::uint64_t>& seed_ids,
                          std::size_t max_results) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::uint64_t> out;
    std::unordered_set<std::uint64_t> seen(seed_ids.begin(), seed_ids.end());
    for (auto id : seed_ids) {
        auto rit = refs_by_chunk_.find(id);
        if (rit == refs_by_chunk_.end()) continue;
        for (const auto& r : rit->second) {
            auto dit = defs_by_name_.find(r.name);
            if (dit == defs_by_name_.end()) continue;
            for (const auto& d : dit->second) {
                if (seen.insert(d.chunk_id).second) {
                    out.push_back(d.chunk_id);
                    if (out.size() >= max_results) return out;
                }
            }
        }
    }
    return out;
}

std::vector<SymbolDef>
SymbolGraph::defs_in_file(const std::string& file_path) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto fit = chunks_by_file_.find(file_path);
    if (fit == chunks_by_file_.end()) return {};
    std::vector<SymbolDef> out;
    for (auto id : fit->second) {
        auto dit = defs_by_chunk_.find(id);
        if (dit == defs_by_chunk_.end()) continue;
        for (const auto& d : dit->second) out.push_back(d);
    }
    return out;
}

std::vector<std::string>
SymbolGraph::files_referencing(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = refs_by_name_.find(name);
    if (it == refs_by_name_.end()) return {};
    std::unordered_set<std::string> uniq;
    for (const auto& r : it->second) uniq.insert(r.file_path);
    return {uniq.begin(), uniq.end()};
}

std::size_t SymbolGraph::definition_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return def_total_;
}

std::size_t SymbolGraph::reference_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ref_total_;
}

} // namespace preprocessor
