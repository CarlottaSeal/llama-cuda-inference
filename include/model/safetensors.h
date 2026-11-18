#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ============================================================
//  Safetensors file format loader
//
//  File layout:
//    [8 bytes]  header_size (uint64, little-endian)
//    [header_size bytes]  JSON header (UTF-8)
//    [remaining bytes]    raw tensor data
//
//  JSON header maps tensor names to:
//    { "dtype": "F16"|"F32"|"BF16",
//      "shape": [dim0, dim1, ...],
//      "data_offsets": [begin, end] }
//
//  The data_offsets are relative to the start of the data
//  section (i.e., after the 8-byte size + header).
//
//  We use mmap for zero-copy access to the data section.
// ============================================================

struct TensorInfo {
    std::string name;
    std::string dtype;           // "F16", "F32", "BF16"
    std::vector<int64_t> shape;
    size_t data_offset;          // byte offset from data section start
    size_t data_size;            // byte count
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();

    // Non-copyable
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Open and parse header. Returns false on error (check error()).
    bool open(const std::string& path);

    // Close and unmap.
    void close();

    // Get tensor info by name. Returns nullptr if not found.
    const TensorInfo* tensor(const std::string& name) const;

    // Get raw pointer to tensor data (in mmap'd region).
    // The pointer is valid as long as the file is open.
    const void* tensor_data(const std::string& name) const;
    const void* tensor_data(const TensorInfo& info) const;

    // List all tensor names
    std::vector<std::string> tensor_names() const;

    // Error message from last failed operation
    const std::string& error() const { return error_; }

private:
    std::string error_;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    size_t data_offset_ = 0;     // byte offset to data section in file
    std::unordered_map<std::string, TensorInfo> tensors_;
};
