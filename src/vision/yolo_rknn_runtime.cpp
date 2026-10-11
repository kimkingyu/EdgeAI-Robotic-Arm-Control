#include "yolo_vision.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#ifndef EDGEAI_HAVE_RKNN
#define EDGEAI_HAVE_RKNN 0
#endif
#if EDGEAI_HAVE_RKNN
#include "rknn_api.h"
#endif

namespace edgeai::yolo {
#if EDGEAI_HAVE_RKNN
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void checked(int rc,const char* operation) {
    if(rc!=RKNN_SUCC) throw std::runtime_error(std::string(operation)+" returned "+std::to_string(rc));
}
template<std::size_t N> std::string bounded_string(const char (&text)[N]) {
    std::size_t n=0; while(n<N && text[n]) ++n; return std::string(text,n);
}
struct Context {
    rknn_context handle=0;
    bool live=false;
    ~Context() { if(live) rknn_destroy(handle); }
    Context()=default;
    Context(const Context&)=delete;
    Context& operator=(const Context&)=delete;
};
TensorInfo tensor_info(const rknn_tensor_attr& a) {
    if(a.n_dims==0 || a.n_dims>RKNN_MAX_DIMS) throw std::runtime_error("Invalid RKNN tensor rank");
    std::uint64_t elements=1;
    for(std::uint32_t j=0;j<a.n_dims;++j) {
        if(!a.dims[j] || elements>std::numeric_limits<std::uint32_t>::max()/a.dims[j])
            throw std::runtime_error("Invalid/overflowed tensor dimensions");
        elements*=a.dims[j];
    }
    if(elements!=a.n_elems) throw std::runtime_error("Tensor element count contradicts dimensions");
    if(a.type!=RKNN_TENSOR_FLOAT32 && a.type!=RKNN_TENSOR_FLOAT16 &&
       a.type!=RKNN_TENSOR_INT8 && a.type!=RKNN_TENSOR_UINT8)
        throw std::runtime_error("Unsupported tensor type (only FP32/FP16/INT8/UINT8)");
    if(a.qnt_type!=RKNN_TENSOR_QNT_NONE && a.qnt_type!=RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
        throw std::runtime_error("Unknown tensor quantization contract");
    if(a.qnt_type==RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
        validate_affine_metadata(get_type_string(a.type),a.scale,a.zp);
    if(a.fmt!=RKNN_TENSOR_NCHW && a.fmt!=RKNN_TENSOR_NHWC && a.fmt!=RKNN_TENSOR_UNDEFINED)
        throw std::runtime_error("Unknown tensor layout contract");
    TensorInfo t; t.index=a.index; t.name=bounded_string(a.name);
    t.format=get_format_string(a.fmt); t.type=get_type_string(a.type);
    t.quantization=get_qnt_type_string(a.qnt_type); t.dims.assign(a.dims,a.dims+a.n_dims);
    t.elements=a.n_elems; t.bytes=a.size; t.width_stride=a.w_stride; t.height_stride=a.h_stride;
    t.bytes_with_stride=a.size_with_stride; t.zero_point=a.zp; t.scale=a.scale; t.fractional_length=a.fl;
    return t;
}
// Caller-owned output buffers remove allocation ownership ambiguity even when outputs_get fails.
struct OutputLease {
    rknn_context ctx;
    std::vector<rknn_output> descriptors;
    bool acquired=false;
    OutputLease(rknn_context c,std::size_t n):ctx(c),descriptors(n){}
    ~OutputLease() { if(acquired) rknn_outputs_release(ctx,static_cast<std::uint32_t>(descriptors.size()),descriptors.data()); }
    void release() {
        const int rc=rknn_outputs_release(ctx,static_cast<std::uint32_t>(descriptors.size()),descriptors.data());
        acquired=false; checked(rc,"rknn_outputs_release");
    }
};
class RknnRuntime final:public Runtime {
    std::vector<std::uint8_t> model_; // declared first: released AFTER the context
    RuntimeInfo info_;
    Context ctx_;
public:
    RknnRuntime(const std::string& path,int core_mask) {
        if(core_mask!=1 && core_mask!=2 && core_mask!=4) throw std::runtime_error("Single worker core mask must be 1, 2 or 4");
        if(!std::filesystem::is_regular_file(path)) throw std::runtime_error("Model must be a readable regular file");
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        const auto size=file.tellg();
        if(!file || size<=0 || static_cast<std::uint64_t>(size)>std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Model is missing, empty, unreadable or exceeds RKNN size limit");
        model_.resize(static_cast<std::size_t>(size)); file.seekg(0);
        if(!file.read(reinterpret_cast<char*>(model_.data()),size)) throw std::runtime_error("Incomplete model read");
        const int initialized=rknn_init(&ctx_.handle,model_.data(),static_cast<std::uint32_t>(model_.size()),0,nullptr);
        // An implementation returning a nonzero handle on failure is still cleaned up.
        ctx_.live=ctx_.handle!=0;
        checked(initialized,"rknn_init"); ctx_.live=true;
        rknn_input_output_num count{};
        checked(rknn_query(ctx_.handle,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count)),"rknn_query(counts)");
        if(count.n_input!=1 || (count.n_output!=1 && count.n_output!=2))
            throw std::runtime_error("Expected exactly one input and one/two outputs");
        rknn_sdk_version version{};
        checked(rknn_query(ctx_.handle,RKNN_QUERY_SDK_VERSION,&version,sizeof(version)),"rknn_query(sdk_version)");
        info_.provider="RKNN synchronous inputs_set/run + caller-owned float outputs";
        info_.sdk_version=bounded_string(version.api_version); info_.driver_version=bounded_string(version.drv_version);
        rknn_tensor_attr input{};
        checked(rknn_query(ctx_.handle,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"rknn_query(input_attr)");
        info_.input=tensor_info(input);
        if(input.n_dims!=4 || input.dims[0]!=1 ||
           (input.fmt!=RKNN_TENSOR_NHWC && input.fmt!=RKNN_TENSOR_NCHW))
            throw std::runtime_error("Input must be explicit static batch=1 NHWC/NCHW");
        const bool nhwc=input.fmt==RKNN_TENSOR_NHWC;
        const auto channels=input.dims[nhwc?3:1];
        info_.input_height=static_cast<int>(input.dims[nhwc?1:2]);
        info_.input_width=static_cast<int>(input.dims[nhwc?2:3]);
        if(channels!=3 || info_.input_width!=640 || info_.input_height!=640)
            throw std::runtime_error("Only static three-channel 640x640 RGB input is supported");
        rknn_input_range range{};
        const int range_rc=rknn_query(ctx_.handle,RKNN_QUERY_INPUT_DYNAMIC_RANGE,&range,sizeof(range));
        if(range_rc==RKNN_SUCC && range.shape_number>1) throw std::runtime_error("Dynamic input shapes are unsupported");
        std::vector<std::vector<std::uint32_t>> dims;
        for(std::uint32_t i=0;i<count.n_output;++i) {
            rknn_tensor_attr attr{}; attr.index=i;
            checked(rknn_query(ctx_.handle,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"rknn_query(output_attr)");
            info_.outputs.push_back(tensor_info(attr)); dims.push_back(info_.outputs.back().dims);
        }
        info_.layout=validate_output_layout(dims);
        checked(rknn_set_core_mask(ctx_.handle,static_cast<rknn_core_mask>(core_mask)),"rknn_set_core_mask");
        info_.core_mask=core_mask;
    }
    const RuntimeInfo& info() const override {return info_;}
    Inference infer(const Input& input) override {
        validate_transform(input.transform);
        if(input.transform.target_width!=info_.input_width || input.transform.target_height!=info_.input_height ||
           input.rgb.size()!=static_cast<std::size_t>(info_.input_width)*info_.input_height*3)
            throw std::runtime_error("RGB input size does not match queried tensor contract");
        Inference result;
        rknn_input descriptor{}; descriptor.index=0;
        descriptor.buf=const_cast<std::uint8_t*>(input.rgb.data());
        descriptor.size=static_cast<std::uint32_t>(input.rgb.size());
        descriptor.pass_through=0; descriptor.type=RKNN_TENSOR_UINT8; descriptor.fmt=RKNN_TENSOR_NHWC;
        const auto run_start=Clock::now();
        checked(rknn_inputs_set(ctx_.handle,1,&descriptor),"rknn_inputs_set");
        checked(rknn_run(ctx_.handle,nullptr),"rknn_run");
        result.inference_call_ms=elapsed(run_start);
        const auto output_start=Clock::now();
        result.outputs.resize(info_.outputs.size());
        OutputLease lease(ctx_.handle,info_.outputs.size());
        for(std::size_t i=0;i<info_.outputs.size();++i) {
            auto& tensor=result.outputs[i]; tensor.dims=info_.outputs[i].dims;
            tensor.values.resize(info_.outputs[i].elements);
            auto& output=lease.descriptors[i]; output.index=static_cast<std::uint32_t>(i);
            output.want_float=1; output.is_prealloc=1; output.buf=tensor.values.data();
            output.size=static_cast<std::uint32_t>(tensor.values.size()*sizeof(float));
        }
        checked(rknn_outputs_get(ctx_.handle,static_cast<std::uint32_t>(lease.descriptors.size()),lease.descriptors.data(),nullptr),
                "rknn_outputs_get");
        lease.acquired=true;
        for(std::size_t i=0;i<result.outputs.size();++i) {
            const auto& out=lease.descriptors[i];
            if(out.buf!=result.outputs[i].values.data() || out.size!=result.outputs[i].values.size()*sizeof(float))
                throw std::runtime_error("RKNN float output buffer contradicts dense tensor contract");
        }
        lease.release(); result.output_ms=elapsed(output_start); return result;
    }
};
} // namespace
#endif
bool sdk_available() noexcept {return EDGEAI_HAVE_RKNN!=0;}
std::unique_ptr<Runtime> create_runtime(const std::string& model,int core_mask) {
#if EDGEAI_HAVE_RKNN
    return std::make_unique<RknnRuntime>(model,core_mask);
#else
    (void)model; (void)core_mask;
    throw std::runtime_error("RKNN SDK/runtime support is disabled; real inference unavailable (no mock)");
#endif
}
} // namespace edgeai::yolo
