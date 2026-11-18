#include "model/tokenizer.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>

// ============================================================
//  Minimal JSON helpers (reused pattern from safetensors)
//  Just enough to parse tokenizer.json
// ============================================================

namespace {

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        i++;
}

bool parse_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    i++;
    out.clear();
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            i++;
            if (i >= s.size()) return false;
            switch (s[i]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'u': {
                    // Parse \uXXXX
                    if (i + 4 >= s.size()) return false;
                    std::string hex = s.substr(i + 1, 4);
                    i += 4;
                    unsigned codepoint = std::stoul(hex, nullptr, 16);
                    if (codepoint < 0x80) {
                        out += (char)codepoint;
                    } else if (codepoint < 0x800) {
                        out += (char)(0xC0 | (codepoint >> 6));
                        out += (char)(0x80 | (codepoint & 0x3F));
                    } else {
                        out += (char)(0xE0 | (codepoint >> 12));
                        out += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                        out += (char)(0x80 | (codepoint & 0x3F));
                    }
                    break;
                }
                default: out += s[i]; break;
            }
        } else {
            out += s[i];
        }
        i++;
    }
    if (i >= s.size()) return false;
    i++; // closing quote
    return true;
}

bool parse_int(const std::string& s, size_t& i, int& out) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    bool neg = false;
    if (s[i] == '-') { neg = true; i++; }
    if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
    int val = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        val = val * 10 + (s[i] - '0');
        i++;
    }
    out = neg ? -val : val;
    return true;
}

void skip_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) return;
    if (s[i] == '"') {
        std::string tmp;
        parse_string(s, i, tmp);
    } else if (s[i] == '{') {
        i++; int d = 1;
        while (i < s.size() && d > 0) {
            if (s[i] == '{') d++;
            else if (s[i] == '}') d--;
            else if (s[i] == '"') { std::string t; parse_string(s, i, t); continue; }
            i++;
        }
    } else if (s[i] == '[') {
        i++; int d = 1;
        while (i < s.size() && d > 0) {
            if (s[i] == '[') d++;
            else if (s[i] == ']') d--;
            else if (s[i] == '"') { std::string t; parse_string(s, i, t); continue; }
            i++;
        }
    } else {
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']'
               && s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t')
            i++;
    }
}

// Find a key in a JSON object (cursor must be right after '{')
// Advances cursor past the key's value
bool find_key(const std::string& s, size_t& i, const std::string& key) {
    while (true) {
        skip_ws(s, i);
        if (i >= s.size() || s[i] == '}') return false;
        if (s[i] == ',') { i++; continue; }

        std::string k;
        if (!parse_string(s, i, k)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') return false;
        i++; // skip :
        skip_ws(s, i);

        if (k == key) return true;
        skip_value(s, i);
    }
}

} // anonymous namespace

// ============================================================
//  Tokenizer implementation
// ============================================================

bool Tokenizer::load(const std::string& path) {
    // Read entire file
    std::ifstream f(path);
    if (!f.is_open()) {
        error_ = "Failed to open: " + path;
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    f.close();

    // Parse vocab from "model" -> "vocab"
    // tokenizer.json structure (simplified):
    // {
    //   "model": {
    //     "vocab": { "token": id, ... },
    //     "merges": [ "a b", "c d", ... ]
    //   },
    //   "added_tokens": [ { "id": ..., "content": ... }, ... ]
    // }

    size_t i = 0;
    skip_ws(json, i);
    if (json[i] != '{') { error_ = "Expected top-level object"; return false; }
    i++;

    // We need to find "model" and "added_tokens" sections
    // First pass: find model
    size_t saved_i = i;

    if (!find_key(json, i, "model")) {
        error_ = "Could not find 'model' key";
        return false;
    }

    // Now at the model object
    if (json[i] != '{') { error_ = "Expected model object"; return false; }
    i++;

    // Find vocab and merges within model
    size_t model_start = i;

    // Find vocab
    if (!find_key(json, i, "vocab")) {
        error_ = "Could not find 'vocab' in model";
        return false;
    }

    // Parse vocab: { "token": id, "token2": id2, ... }
    if (json[i] != '{') { error_ = "Expected vocab object"; return false; }
    i++;

    while (true) {
        skip_ws(json, i);
        if (i >= json.size() || json[i] == '}') { i++; break; }
        if (json[i] == ',') { i++; continue; }

        std::string token;
        if (!parse_string(json, i, token)) { error_ = "Failed to parse vocab token"; return false; }
        skip_ws(json, i);
        if (json[i] != ':') { error_ = "Expected ':' in vocab"; return false; }
        i++;
        int id;
        if (!parse_int(json, i, id)) { error_ = "Failed to parse vocab id"; return false; }

        token_to_id_[token] = id;
        if (id >= (int)id_to_token_.size()) {
            id_to_token_.resize(id + 1);
        }
        id_to_token_[id] = token;
    }

    printf("  Loaded %zu vocab entries\n", token_to_id_.size());

    // Find merges — go back to model start and search
    i = model_start;
    if (find_key(json, i, "merges")) {
        if (json[i] != '[') { error_ = "Expected merges array"; return false; }
        i++;

        int rank = 0;
        while (true) {
            skip_ws(json, i);
            if (i >= json.size() || json[i] == ']') { i++; break; }
            if (json[i] == ',') { i++; continue; }

            std::string merge_str;
            if (!parse_string(json, i, merge_str)) break;

            // Split "a b" into a and b (split on first space)
            auto space_pos = merge_str.find(' ');
            if (space_pos != std::string::npos) {
                Merge m;
                m.a = merge_str.substr(0, space_pos);
                m.b = merge_str.substr(space_pos + 1);
                merges_.push_back(m);
                merge_ranks_[merge_str] = rank++;
            }
        }
        printf("  Loaded %zu merges\n", merges_.size());
    }

    // Find added_tokens (special tokens like BOS, EOS)
    i = saved_i;
    if (find_key(json, i, "added_tokens")) {
        if (json[i] == '[') {
            i++;
            while (true) {
                skip_ws(json, i);
                if (i >= json.size() || json[i] == ']') { i++; break; }
                if (json[i] == ',') { i++; continue; }

                if (json[i] == '{') {
                    i++;
                    int id = -1;
                    std::string content;
                    // Parse object fields
                    while (true) {
                        skip_ws(json, i);
                        if (i >= json.size() || json[i] == '}') { i++; break; }
                        if (json[i] == ',') { i++; continue; }

                        std::string key;
                        if (!parse_string(json, i, key)) break;
                        skip_ws(json, i);
                        if (json[i] != ':') break;
                        i++;
                        skip_ws(json, i);

                        if (key == "id") {
                            parse_int(json, i, id);
                        } else if (key == "content") {
                            parse_string(json, i, content);
                        } else {
                            skip_value(json, i);
                        }
                    }

                    if (id >= 0 && !content.empty()) {
                        token_to_id_[content] = id;
                        if (id >= (int)id_to_token_.size()) {
                            id_to_token_.resize(id + 1);
                        }
                        id_to_token_[id] = content;

                        if (content == "<|begin_of_text|>") bos_id_ = id;
                        if (content == "<|end_of_text|>") eos_id_ = id;
                    }
                } else {
                    skip_value(json, i);
                }
            }
        }
    }

    printf("  BOS id: %d, EOS id: %d\n", bos_id_, eos_id_);
    printf("  Total vocab size: %d\n", vocab_size());

    return true;
}

// ============================================================
//  BPE encoding
// ============================================================

void Tokenizer::apply_bpe(std::vector<std::string>& tokens) const {
    // Repeatedly find and apply the highest-priority merge
    while (tokens.size() > 1) {
        // Find the merge with the lowest rank (highest priority)
        int best_rank = INT32_MAX;
        int best_idx = -1;

        for (int i = 0; i < (int)tokens.size() - 1; i++) {
            std::string pair = tokens[i] + " " + tokens[i + 1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;  // no more merges applicable

        // Apply the merge: combine tokens[best_idx] and tokens[best_idx+1]
        tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;

    if (add_bos && bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }

    if (text.empty()) return ids;

    // Byte-level BPE: start with individual bytes as initial tokens
    // Each byte maps to a single-character string in the vocab
    std::vector<std::string> tokens;
    for (unsigned char c : text) {
        tokens.push_back(std::string(1, (char)c));
    }

    // Apply BPE merges
    apply_bpe(tokens);

    // Convert tokens to IDs
    for (auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown token — fall back to byte-level encoding
            for (unsigned char c : tok) {
                std::string byte_tok(1, (char)c);
                auto bit = token_to_id_.find(byte_tok);
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
                // If even the byte isn't in vocab, skip it
            }
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        result += decode_token(id);
    }
    return result;
}

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || id >= (int)id_to_token_.size()) return "";

    const std::string& tok = id_to_token_[id];

    // Skip special tokens in output
    if (!tok.empty() && tok[0] == '<' && tok.back() == '>') {
        return "";
    }

    return tok;
}
