// Metadata-only RKNN probe. No camera, controller, I2C, or inference calls.
// Build against the official pinned rknn_api.h and the deployed librknnrt.
#include "rknn_api.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                              << std::setfill('0') << static_cast<unsigned>(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
struct Context {
    rknn_context value = 0;
    bool initialized = false;
    ~Context() { if (initialized) rknn_destroy(value); }
};
void require(int result, const char* operation) {
    if (result != RKNN_SUCC)
        throw std::runtime_error(std::string(operation) + " returned " + std::to_string(result));
}
void attributes(std::ostream& out, rknn_context ctx, rknn_query_cmd command, uint32_t count) {
    if (count > 1024) throw std::runtime_error("Unreasonable tensor count");
    out << '[';
    for (uint32_t i = 0; i < count; ++i) {
        rknn_tensor_attr a{};
        a.index = i;
        require(rknn_query(ctx, command, &a, sizeof(a)), "rknn_query(tensor_attr)");
        if (a.n_dims > RKNN_MAX_DIMS) throw std::runtime_error("Invalid tensor rank");
        // Do not trust the runtime to terminate fixed-size name buffers.
        std::size_t name_size = 0;
        while (name_size < sizeof(a.name) && a.name[name_size]) ++name_size;
        if (i) out << ',';
        out << "{\"index\":" << a.index << ",\"name\":" << quote(std::string(a.name, name_size))
            << ",\"dims\":[";
        for (uint32_t j = 0; j < a.n_dims; ++j) { if (j) out << ','; out << a.dims[j]; }
        out << "],\"format\":" << quote(get_format_string(a.fmt))
            << ",\"type\":" << quote(get_type_string(a.type))
            << ",\"quantization\":" << static_cast<int>(a.qnt_type)
            << ",\"zero_point\":" << a.zp << ",\"scale\":";
        if (std::isfinite(a.scale)) out << std::setprecision(9) << a.scale;
        else out << "null";
        out << ",\"elements\":" << a.n_elems << ",\"bytes\":" << a.size
            << ",\"width_stride\":" << a.w_stride
            << ",\"bytes_with_stride\":" << a.size_with_stride << '}';
    }
    out << ']';
}
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: inspect_rknn_contract MODEL.rknn REPORT.json\n";
        return 2;
    }
    try {
        std::ifstream model(argv[1], std::ios::binary);
        if (!model) throw std::runtime_error("Model file not readable");
        Context ctx;
        require(rknn_init(&ctx.value, argv[1], 0, 0, nullptr), "rknn_init");
        ctx.initialized = true;
        rknn_input_output_num io{};
        rknn_sdk_version version{};
        require(rknn_query(ctx.value, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query counts");
        require(rknn_query(ctx.value, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), "query versions");
        version.api_version[sizeof(version.api_version) - 1] = '\0';
        version.drv_version[sizeof(version.drv_version) - 1] = '\0';
        std::ostringstream result;
        result << "{\"schema_version\":1,\"model_path\":" << quote(argv[1])
               << ",\"runtime\":" << quote(version.api_version)
               << ",\"driver\":" << quote(version.drv_version)
               << ",\"metadata_only\":true,\"inputs\":";
        attributes(result, ctx.value, RKNN_QUERY_INPUT_ATTR, io.n_input);
        result << ",\"outputs\":";
        attributes(result, ctx.value, RKNN_QUERY_OUTPUT_ATTR, io.n_output);
        result << "}\n";
        // Write only after every query succeeded; invalid metadata is not a report.
        std::ofstream report(argv[2], std::ios::binary);
        if (!report || !(report << result.str())) throw std::runtime_error("Cannot write report");
        std::cout << "RKNN metadata: inputs=" << io.n_input << " outputs=" << io.n_output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Contract probe failed: " << error.what() << '\n';
        return 1;
    }
}
