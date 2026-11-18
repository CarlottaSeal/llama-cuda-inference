#include "model/safetensors.h"

#include <fstream>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ============================================================
//  Read-only file mapping: mmap on POSIX, MapViewOfFile on Windows
// ============================================================

static void* map_file(const std::string& path, size_t& size, std::string& err) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { err = "Failed to open file: " + path; return nullptr; }

    LARGE_INTEGER li;
    if (!GetFileSizeEx(file, &li)) {
        CloseHandle(file);
        err = "Failed to stat file: " + path;
        return nullptr;
    }
    size = (size_t)li.QuadPart;
    if (size == 0) { CloseHandle(file); err = "Empty file: " + path; return nullptr; }

    // The view keeps the mapping alive; both handles can be closed now
    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (!mapping) { err = "CreateFileMapping failed"; return nullptr; }
    void* ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (!ptr) err = "MapViewOfFile failed";
    return ptr;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "Failed to open file: " + path; return nullptr; }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        ::close(fd);
        err = "Failed to stat file: " + path;
        return nullptr;
    }
    size = st.st_size;
    if (size == 0) { ::close(fd); err = "Empty file: " + path; return nullptr; }

    void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) { err = "mmap failed"; return nullptr; }
    return ptr;
#endif
}

static void unmap_file(void* ptr, size_t size) {
#ifdef _WIN32
    (void)size;
    UnmapViewOfFile(ptr);
#else
    munmap(ptr, size);
#endif
}

// ============================================================
//  JSON parsing — minimal hand-rolled parser
//
//  We only need to parse the safetensors header format:
//    { "tensor_name": { "dtype": "F16", "shape": [d0, d1],
//                       "data_offsets": [begin, end] },
//      "__metadata__": { ... } }
//
//  This avoids the nlohmann/json dependency while still being
//  robust enough for safetensors headers.
// ============================================================

namespace {

// Skip whitespace
void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        i++;
}

// Parse a JSON string (expects opening quote at s[i])
bool parse_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    i++; // skip opening quote
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
                default: out += s[i]; break;
            }
        } else {
            out += s[i];
        }
        i++;
    }
    if (i >= s.size()) return false;
    i++; // skip closing quote
    return true;
}

// Parse a JSON integer
bool parse_int64(const std::string& s, size_t& i, int64_t& out) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    bool neg = false;
    if (s[i] == '-') { neg = true; i++; }
    if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
    int64_t val = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        val = val * 10 + (s[i] - '0');
        i++;
    }
    out = neg ? -val : val;
    return true;
}

// Skip a JSON value (string, number, object, array, bool, null) recursively
void skip_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) return;
    if (s[i] == '"') {
        std::string tmp;
        parse_string(s, i, tmp);
    } else if (s[i] == '{') {
        i++;
        int depth = 1;
        while (i < s.size() && depth > 0) {
            if (s[i] == '{') depth++;
            else if (s[i] == '}') depth--;
            else if (s[i] == '"') { std::string tmp; parse_string(s, i, tmp); continue; }
            i++;
        }
    } else if (s[i] == '[') {
        i++;
        int depth = 1;
        while (i < s.size() && depth > 0) {
            if (s[i] == '[') depth++;
            else if (s[i] == ']') depth--;
            else if (s[i] == '"') { std::string tmp; parse_string(s, i, tmp); continue; }
            i++;
        }
    } else {
        // number, bool, null — advance to next delimiter
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']'
               && s[i] != ' ' && s[i] != '\n' && s[i] != '\r' && s[i] != '\t')
            i++;
    }
}

// Parse the tensor info object: { "dtype": ..., "shape": [...], "data_offsets": [...] }
bool parse_tensor_info(const std::string& s, size_t& i, TensorInfo& info) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{') return false;
    i++; // skip {

    while (true) {
        skip_ws(s, i);
        if (i >= s.size()) return false;
        if (s[i] == '}') { i++; break; }
        if (s[i] == ',') { i++; continue; }

        std::string key;
        if (!parse_string(s, i, key)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') return false;
        i++; // skip :
        skip_ws(s, i);

        if (key == "dtype") {
            if (!parse_string(s, i, info.dtype)) return false;
        } else if (key == "shape") {
            // Parse array of ints
            if (s[i] != '[') return false;
            i++;
            info.shape.clear();
            while (true) {
                skip_ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ']') { i++; break; }
                if (s[i] == ',') { i++; continue; }
                int64_t val;
                if (!parse_int64(s, i, val)) return false;
                info.shape.push_back(val);
            }
        } else if (key == "data_offsets") {
            // Parse [begin, end]
            if (s[i] != '[') return false;
            i++;
            int64_t begin_off, end_off;
            if (!parse_int64(s, i, begin_off)) return false;
            skip_ws(s, i);
            if (s[i] == ',') i++;
            if (!parse_int64(s, i, end_off)) return false;
            skip_ws(s, i);
            if (s[i] == ']') i++;
            info.data_offset = (size_t)begin_off;
            info.data_size = (size_t)(end_off - begin_off);
        } else {
            skip_value(s, i);
        }
    }
    return true;
}

} // anonymous namespace

// ============================================================
//  SafetensorsFile implementation
// ============================================================

SafetensorsFile::~SafetensorsFile() {
    close();
}

void SafetensorsFile::close() {
    if (mmap_ptr_) {
        unmap_file(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
        mmap_size_ = 0;
    }
    tensors_.clear();
}

bool SafetensorsFile::open(const std::string& path) {
    close();

    // Map the entire file read-only
    mmap_ptr_ = map_file(path, mmap_size_, error_);
    if (!mmap_ptr_) {
        mmap_size_ = 0;
        return false;
    }

    if (mmap_size_ < 8) {
        error_ = "File too small for safetensors header";
        close();
        return false;
    }

    // Read header size (first 8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    std::memcpy(&header_size, mmap_ptr_, 8);

    if (8 + header_size > mmap_size_) {
        error_ = "Header size exceeds file size";
        close();
        return false;
    }

    data_offset_ = 8 + header_size;

    // Parse JSON header
    std::string json_str((const char*)mmap_ptr_ + 8, header_size);

    size_t i = 0;
    skip_ws(json_str, i);
    if (i >= json_str.size() || json_str[i] != '{') {
        error_ = "Invalid JSON header: expected '{'";
        close();
        return false;
    }
    i++; // skip {

    while (true) {
        skip_ws(json_str, i);
        if (i >= json_str.size()) break;
        if (json_str[i] == '}') break;
        if (json_str[i] == ',') { i++; continue; }

        // Parse tensor name
        std::string tensor_name;
        if (!parse_string(json_str, i, tensor_name)) {
            error_ = "Failed to parse tensor name in header";
            close();
            return false;
        }

        skip_ws(json_str, i);
        if (i >= json_str.size() || json_str[i] != ':') {
            error_ = "Expected ':' after tensor name";
            close();
            return false;
        }
        i++; // skip :
        skip_ws(json_str, i);

        // Skip __metadata__
        if (tensor_name == "__metadata__") {
            skip_value(json_str, i);
            continue;
        }

        // Parse tensor info
        TensorInfo info;
        info.name = tensor_name;
        if (!parse_tensor_info(json_str, i, info)) {
            error_ = "Failed to parse tensor info for: " + tensor_name;
            close();
            return false;
        }

        tensors_[tensor_name] = std::move(info);
    }

    return true;
}

const TensorInfo* SafetensorsFile::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

const void* SafetensorsFile::tensor_data(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return tensor_data(it->second);
}

const void* SafetensorsFile::tensor_data(const TensorInfo& info) const {
    if (!mmap_ptr_) return nullptr;
    return (const char*)mmap_ptr_ + data_offset_ + info.data_offset;
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (auto& [k, v] : tensors_) {
        names.push_back(k);
    }
    return names;
}
