#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ============================================================
//  Minimal BPE Tokenizer
//
//  Reads HuggingFace tokenizer.json format (tiktoken-style BPE).
//
//  Llama 3's tokenizer uses byte-level BPE:
//    - Vocabulary maps byte sequences to token IDs
//    - Merges define which byte-pair combinations form tokens
//    - Special tokens: <|begin_of_text|> (128000),
//      <|end_of_text|> (128001), etc.
//
//  Encode: string → bytes → greedy BPE merge → token IDs
//  Decode: token IDs → byte sequences → string
// ============================================================

class Tokenizer {
public:
    // Load from tokenizer.json
    bool load(const std::string& path);

    // Encode text to token IDs (prepends BOS by default)
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int>& ids) const;

    // Decode a single token
    std::string decode_token(int id) const;

    // Vocabulary size
    int vocab_size() const { return (int)id_to_token_.size(); }

    // Special token IDs
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

    const std::string& error() const { return error_; }

private:
    std::string error_;

    // Vocabulary: token string <-> ID
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    // BPE merges: ordered list of (token_a, token_b) pairs
    // Lower index = higher priority
    struct Merge {
        std::string a, b;
    };
    std::vector<Merge> merges_;

    // Merge priority lookup: "a b" -> rank (lower = merge first)
    std::unordered_map<std::string, int> merge_ranks_;

    // Special tokens
    int bos_id_ = -1;
    int eos_id_ = -1;

    // Apply BPE merges to a list of tokens
    void apply_bpe(std::vector<std::string>& tokens) const;
};
