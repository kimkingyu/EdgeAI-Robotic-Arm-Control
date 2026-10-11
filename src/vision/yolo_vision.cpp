#include "yolo_vision.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#ifndef EDGEAI_HAVE_OPENCV
#define EDGEAI_HAVE_OPENCV 1
#endif
#if EDGEAI_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace edgeai::yolo {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
const char* classes[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};
int round_even(double x) {
    const double floor_value = std::floor(x);
    int result = static_cast<int>(floor_value);
    if (x - floor_value > .5 || (x - floor_value == .5 && result % 2)) ++result;
    return result;
}
struct Candidate { std::array<double, 4> box; double score; int cls; std::size_t anchor; };
double iou(const Candidate& a, const Candidate& b) {
    const double area_a = std::max(0.0, a.box[2]-a.box[0]) * std::max(0.0, a.box[3]-a.box[1]);
    const double area_b = std::max(0.0, b.box[2]-b.box[0]) * std::max(0.0, b.box[3]-b.box[1]);
    const double intersection = std::max(0.0, std::min(a.box[2],b.box[2])-std::max(a.box[0],b.box[0])) *
        std::max(0.0, std::min(a.box[3],b.box[3])-std::max(a.box[1],b.box[1]));
    const double total = area_a + area_b - intersection;
    return total > 0 ? intersection / std::max(total, 1e-9) : 0;
}
int bounded_integer(const std::string& text, bool zero, const std::string& key) {
    require(!text.empty() && text.size() <= 7 &&
            std::all_of(text.begin(),text.end(),[](char c){return c >= '0' && c <= '9';}),
            key + " requires an unsigned finite integer");
    const auto value = std::stoul(text);
    require(value <= 1000000 && (zero || value > 0), key + " out of range (maximum 1000000)");
    return static_cast<int>(value);
}
} // namespace
Transform make_transform(int sw, int sh, int tw, int th, const std::string& mode) {
    require(sw > 0 && sh > 0 && tw > 0 && th > 0, "Geometry dimensions must be positive");
    require(mode == "resize" || mode == "letterbox", "Unknown geometry mode");
    Transform t;
    t.mode=mode; t.source_width=sw; t.source_height=sh; t.target_width=tw; t.target_height=th;
    if (mode == "resize") { t.resized_width=tw; t.resized_height=th; }
    else {
        const double scale = std::min(static_cast<double>(tw)/sw,static_cast<double>(th)/sh);
        t.resized_width=static_cast<int>(std::floor(sw*scale+.5));
        t.resized_height=static_cast<int>(std::floor(sh*scale+.5));
        require(t.resized_width>0 && t.resized_height>0, "Rounded letterbox dimension is zero");
        t.pad_left=(tw-t.resized_width)/2; t.pad_top=(th-t.resized_height)/2;
    }
    t.scale_x=static_cast<double>(t.resized_width)/sw;
    t.scale_y=static_cast<double>(t.resized_height)/sh;
    return t;
}
void validate_transform(const Transform& t) {
    const auto e=make_transform(t.source_width,t.source_height,t.target_width,t.target_height,t.mode);
    require(t.resized_width==e.resized_width && t.resized_height==e.resized_height &&
        t.pad_left==e.pad_left && t.pad_top==e.pad_top && std::isfinite(t.scale_x) &&
        std::isfinite(t.scale_y) && std::abs(t.scale_x-e.scale_x)<=e.scale_x*1e-12 &&
        std::abs(t.scale_y-e.scale_y)<=e.scale_y*1e-12, "Inconsistent transform contract");
}
Input preprocess(const cv::Mat& bgr, int width, int height, const std::string& mode) {
#if EDGEAI_HAVE_OPENCV
    require(!bgr.empty() && bgr.type()==CV_8UC3, "Expected a nonempty uint8 BGR image");
    Input input; input.transform=make_transform(bgr.cols,bgr.rows,width,height,mode);
    const auto& t=input.transform;
    cv::Mat resized, padded, rgb;
    cv::resize(bgr,resized,cv::Size(t.resized_width,t.resized_height),0,0,cv::INTER_LINEAR);
    if (mode=="letterbox") cv::copyMakeBorder(resized,padded,t.pad_top,
        height-t.resized_height-t.pad_top,t.pad_left,width-t.resized_width-t.pad_left,
        cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    else padded=resized;
    cv::cvtColor(padded,rgb,cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb=rgb.clone();
    input.rgb.assign(rgb.data,rgb.data+rgb.total()*rgb.elemSize());
    return input;
#else
    (void)bgr; (void)width; (void)height; (void)mode;
    throw std::runtime_error("OpenCV support is disabled; preprocessing unavailable (no mock)");
#endif
}
void validate_affine_metadata(const std::string& type, double scale, int zero_point) {
    require(std::isfinite(scale) && scale>0, "Invalid affine scale: must be finite and positive");
    if (type=="FP16" || type=="FP32") {
        // RKNN 2.3.2 reports the verified floating-point model with identity affine fields.
        require(scale==1.0 && zero_point==0, "Floating-point affine metadata must be identity (scale=1, zp=0)");
    } else if (type=="INT8") {
        require(zero_point>=-128 && zero_point<=127, "INT8 affine zero point is outside [-128,127]");
    } else if (type=="UINT8") {
        require(zero_point>=0 && zero_point<=255, "UINT8 affine zero point is outside [0,255]");
    } else throw std::runtime_error("Unsupported affine tensor type");
}
std::string validate_output_layout(const std::vector<std::vector<std::uint32_t>>& dims) {
    require(dims.size()==1 || dims.size()==2,"Expected one or two YOLO output tensors");
    for (const auto& d:dims) require(d.size()==3 && d[0]==1 && d[2]==8400,
        "Only dense channel-major [1,C,8400] static outputs are supported");
    if (dims.size()==1) { require(dims[0][1]==84,"Single output requires COCO80 [1,84,8400]"); return "single_84x8400"; }
    require((dims[0][1]==4 && dims[1][1]==80) || (dims[0][1]==80 && dims[1][1]==4),
        "Split output requires [1,4,8400] + [1,80,8400]");
    return "split_4x8400_80x8400";
}
std::vector<Detection> decode(const std::vector<Tensor>& outputs, const Transform& t,
                              double confidence, double nms, std::size_t max_detections) {
    validate_transform(t);
    require(std::isfinite(confidence) && confidence>=0 && confidence<=1 &&
        std::isfinite(nms) && nms>=0 && nms<=1 && max_detections>0,"Invalid decode thresholds or limit");
    std::vector<std::vector<std::uint32_t>> dims;
    for (const auto& output:outputs) dims.push_back(output.dims);
    validate_output_layout(dims);
    for (const auto& output:outputs) {
        require(output.values.size()==static_cast<std::size_t>(output.dims[1])*8400,"Output buffer size mismatch");
        require(std::all_of(output.values.begin(),output.values.end(),[](float x){return std::isfinite(x);}),
                "Non-finite YOLO output (NaN/Inf); rejecting the entire frame");
    }
    const Tensor* coords=&outputs[0]; const Tensor* scores=&outputs[0]; std::size_t offset=4*8400;
    if (outputs.size()==2) {
        coords=outputs[0].dims[1]==4?&outputs[0]:&outputs[1];
        scores=outputs[0].dims[1]==80?&outputs[0]:&outputs[1]; offset=0;
    }
    std::vector<Candidate> candidates;
    for (std::size_t a=0;a<8400;++a) {
        int cls=0; double score=scores->values[offset+a];
        for (int c=1;c<80;++c) if (scores->values[offset+c*8400+a]>score) {
            cls=c; score=scores->values[offset+c*8400+a];
        }
        if (score<confidence) continue;
        // Match NumPy's float32 cxcywh->xyxy arithmetic before float64 inverse mapping.
        const float cx=coords->values[a],cy=coords->values[8400+a];
        const float w=coords->values[16800+a],h=coords->values[25200+a];
        const std::array<float,4> network={cx-w/2,cy-h/2,cx+w/2,cy+h/2};
        require(std::all_of(network.begin(),network.end(),[](float x){return std::isfinite(x);}),
                "Non-finite cxcywh-to-xyxy coordinates; rejecting the entire frame");
        candidates.push_back({{network[0],network[1],network[2],network[3]},score,cls,a});
    }
    // Network-space, class-aware NMS BEFORE clipping, matching the Python contract.
    // Equal scores deterministically use ascending class, then descending original anchor.
    std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){
        if(a.score!=b.score) return a.score>b.score;
        if(a.cls!=b.cls) return a.cls<b.cls;
        return a.anchor>b.anchor;
    });
    std::vector<Candidate> kept;
    for (const auto& c:candidates) {
        bool suppress=false;
        for (const auto& k:kept) if (c.cls==k.cls && iou(c,k)>nms) {suppress=true;break;}
        if (!suppress) kept.push_back(c);
        if (kept.size()==max_detections) break;
    }
    std::vector<Detection> result;
    for (const auto& c:kept) {
        Detection d; d.class_id=c.cls; d.class_name=classes[c.cls]; d.score=c.score;
        for (int j=0;j<4;++j) {
            const bool x=j%2==0;
            const double mapped=(c.box[j]-(x?t.pad_left:t.pad_top))/(x?t.scale_x:t.scale_y);
            require(std::isfinite(mapped),"Non-finite inverse-mapped coordinates");
            d.source_box[j]=std::clamp(mapped,0.0,static_cast<double>((x?t.source_width:t.source_height)-1));
            d.bbox[j]=round_even(d.source_box[j]);
        }
        if(d.source_box[2]<=d.source_box[0] || d.source_box[3]<=d.source_box[1]) continue;
        d.center={round_even((d.source_box[0]+d.source_box[2])/2),round_even((d.source_box[1]+d.source_box[3])/2)};
        result.push_back(d);
    }
    return result;
}
const char* usage() noexcept {
    return "edge_yolo_vision --model MODEL (--image FILE | --image-list FILE | --video FILE) "
        "--report FILE [--workers 1|3] [--geometry resize|letterbox] [--iterations N] "
        "[--warmup N] [--queue-capacity N] [--overwrite]\n"
        "Images: N passes over the input list. Video: at most N frames, EOF ends a nonempty run. "
        "Warmup is per worker. Only regular video files, never camera devices.\n";
}
Options parse_options(const std::vector<std::string>& arguments) {
    Options o; std::set<std::string> seen;
    for(std::size_t i=0;i<arguments.size();++i) {
        const auto& key=arguments[i]; require(seen.insert(key).second,"Duplicate option: "+key);
        if(key=="--help") {o.help=true;continue;}
        if(key=="--overwrite") {o.overwrite=true;continue;}
        require(i+1<arguments.size(),"Missing option value: "+key);
        const auto& value=arguments[++i];
        require(!value.empty() && value.rfind("--",0)!=0,"Missing option value: "+key);
        if(key=="--model") o.model=value;
        else if(key=="--image") o.image=value;
        else if(key=="--image-list") o.image_list=value;
        else if(key=="--video") o.video=value;
        else if(key=="--report") o.report=value;
        else if(key=="--geometry") o.geometry=value;
        else if(key=="--workers") o.workers=bounded_integer(value,false,key);
        else if(key=="--iterations") o.iterations=bounded_integer(value,false,key);
        else if(key=="--warmup") o.warmup=bounded_integer(value,true,key);
        else if(key=="--queue-capacity") o.queue_capacity=bounded_integer(value,false,key);
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(o.help) return o;
    require(!o.model.empty() && !o.report.empty(),"--model and --report are required");
    require((!o.image.empty())+(!o.image_list.empty())+(!o.video.empty())==1,"Select exactly one file input mode");
    require(o.workers==1 || o.workers==3,"--workers must be 1 or 3");
    require(o.geometry=="resize" || o.geometry=="letterbox","Invalid --geometry");
    require(o.queue_capacity<=1024,"--queue-capacity maximum is 1024 decoded frames");
    return o;
}
std::string json_quote(const std::string& text) {
    std::ostringstream out; out.imbue(std::locale::classic()); out << '"';
    for(unsigned char c:text) {
        if(c=='"' || c=='\\') out << '\\' << static_cast<char>(c);
        else if(c<0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
        else out << static_cast<char>(c);
    }
    out << '"'; return out.str();
}
std::string json_number(double value) {
    if(!std::isfinite(value)) return "null";
    std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(17) << value; return out.str();
}
std::string transform_json(const Transform& t) {
    validate_transform(t);
    std::ostringstream o; o.imbue(std::locale::classic());
    o << "{\"mode\":" << json_quote(t.mode) << ",\"source_width\":" << t.source_width
      << ",\"source_height\":" << t.source_height << ",\"target_width\":" << t.target_width
      << ",\"target_height\":" << t.target_height << ",\"resized_width\":" << t.resized_width
      << ",\"resized_height\":" << t.resized_height << ",\"pad_left\":" << t.pad_left
      << ",\"pad_top\":" << t.pad_top << ",\"scale_x\":" << json_number(t.scale_x)
      << ",\"scale_y\":" << json_number(t.scale_y) << '}'; return o.str();
}
} // namespace edgeai::yolo
