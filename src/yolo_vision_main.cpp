#include "yolo_vision.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#ifndef EDGEAI_HAVE_OPENCV
#define EDGEAI_HAVE_OPENCV 1
#endif
#if EDGEAI_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace {
using namespace edgeai::yolo;
using Clock=std::chrono::steady_clock;
namespace fs=std::filesystem;
constexpr double missing=std::numeric_limits<double>::quiet_NaN();
static_assert(std::atomic<bool>::is_always_lock_free,"Signal stop flag must be lock-free");
std::atomic<bool> interrupted{false};
void signal_handler(int) {interrupted.store(true,std::memory_order_relaxed);}
[[maybe_unused]] double milliseconds(Clock::time_point a,Clock::time_point b=Clock::now()) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
[[maybe_unused]] void check_stop() { if(interrupted) throw std::runtime_error("Interrupted by signal"); }
struct ReportFile {
    std::FILE* file=nullptr;
    explicit ReportFile(const std::string& path,bool overwrite) {
        if(fs::exists(path) && !fs::is_regular_file(path))
            throw std::runtime_error("Report destination must be a regular file, never a device/directory");
        if(!fs::is_directory(fs::absolute(path).parent_path()))
            throw std::runtime_error("Report parent directory does not exist");
        const int flags=O_WRONLY|O_CREAT|(overwrite?O_TRUNC:O_EXCL);
#ifdef _WIN32
        const int fd=_open(path.c_str(),flags|_O_BINARY,_S_IREAD|_S_IWRITE);
        if(fd>=0) { file=_fdopen(fd,"wb"); if(!file) _close(fd); }
#else
        const int fd=::open(path.c_str(),flags,0644);
        if(fd>=0) { file=::fdopen(fd,"wb"); if(!file) ::close(fd); }
#endif
        if(!file) throw std::runtime_error("Cannot create report (existing file requires --overwrite): "+path+
            "; errno="+std::to_string(errno));
    }
    ~ReportFile() {if(file) std::fclose(file);}
    void write(const std::string& text) {
        if(std::fwrite(text.data(),1,text.size(),file)!=text.size() || std::fflush(file)!=0)
            throw std::runtime_error("Report write/flush failed");
        auto* closing=file; file=nullptr;
        if(std::fclose(closing)!=0) throw std::runtime_error("Report close failed");
    }
};
void no_report_alias(const Options& o,const std::vector<std::string>& images) {
    const auto report=fs::weakly_canonical(fs::absolute(o.report));
    std::vector<std::string> inputs={o.model,o.image_list,o.video};
    inputs.insert(inputs.end(),images.begin(),images.end());
    for(const auto& input:inputs) if(!input.empty()) {
        if(report==fs::weakly_canonical(fs::absolute(input)) ||
           (fs::exists(report) && fs::exists(input) && fs::equivalent(report,input)))
            throw std::runtime_error("Report must not overwrite any input/model file");
    }
}
std::vector<std::string> image_paths(const Options& o) {
    if(!o.image.empty()) return {fs::absolute(o.image).lexically_normal().string()};
    if(o.image_list.empty()) return {};
    if(!fs::is_regular_file(o.image_list)) throw std::runtime_error("Image list must be a regular file");
    std::ifstream file(o.image_list); if(!file) throw std::runtime_error("Cannot read image list");
    const auto parent=fs::absolute(o.image_list).parent_path();
    std::vector<std::string> paths; std::string line;
    while(std::getline(file,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const auto first=line.find_first_not_of(" \t"); if(first==std::string::npos) continue;
        const auto last=line.find_last_not_of(" \t"); line=line.substr(first,last-first+1);
        fs::path p(line); if(p.is_relative()) p=parent/p;
        paths.push_back(p.lexically_normal().string());
        if(paths.size()>1000000) throw std::runtime_error("Image list exceeds bounded frame budget");
    }
    if(file.bad() || paths.empty()) throw std::runtime_error("Empty/unreadable image list");
    if(paths.size()>1000000/static_cast<std::size_t>(o.iterations))
        throw std::runtime_error("Image list times iterations exceeds 1000000 frame budget");
    return paths;
}
std::string tensor_json(const TensorInfo& t) {
    std::ostringstream s; s.imbue(std::locale::classic());
    s << "{\"index\":" << t.index << ",\"name\":" << json_quote(t.name)
      << ",\"format\":" << json_quote(t.format) << ",\"type\":" << json_quote(t.type)
      << ",\"quantization\":" << json_quote(t.quantization) << ",\"dims\":[";
    for(std::size_t i=0;i<t.dims.size();++i) {if(i) s << ','; s << t.dims[i];}
    s << "],\"elements\":" << t.elements << ",\"bytes\":" << t.bytes
      << ",\"width_stride\":" << t.width_stride << ",\"height_stride\":" << t.height_stride
      << ",\"bytes_with_stride\":" << t.bytes_with_stride << ",\"zero_point\":" << t.zero_point
      << ",\"fractional_length\":" << t.fractional_length << ",\"scale\":" << json_number(t.scale) << '}';
    return s.str();
}
struct FrameResult {
    std::uint64_t id=0;
    std::string source,status="cancelled",error;
    int worker=-1;
    double decode_ms=missing,producer_wait_ms=missing,queue_ms=missing,preprocess_ms=missing;
    double inference_call_ms=missing,output_and_nms_ms=missing,total_wall_ms=missing;
    Transform transform;
    std::vector<Detection> detections;
};
struct Worker {
    std::unique_ptr<Runtime> runtime;
    RuntimeInfo metadata;
    double initialization_ms=missing,warmup_wall_ms=missing,first_call_wall_ms=missing;
    int warmup_completed=0;
    std::string error;
    std::vector<FrameResult> results;
};
struct Run {
    std::string error;
    double startup_decode_ms=missing,measurement_wall_ms=missing;
    bool video_eof=false;
    QueueStats queue;
    std::vector<Worker> workers;
    std::vector<FrameResult> attempts,frames;
};
std::string percentile(std::vector<double> values,double p) {
    if(values.empty()) return "null";
    std::sort(values.begin(),values.end());
    // Explicit nearest-rank percentile, never summed across stages.
    const auto rank=static_cast<std::size_t>(std::ceil(p*values.size()));
    return json_number(values[std::max<std::size_t>(1,rank)-1]);
}
std::string render(const Options& o,const Run& run) {
    std::ostringstream s; s.imbue(std::locale::classic());
    std::size_t completed=0,failed=0,cancelled=0;
    for(const auto& f:run.frames) {if(f.status=="ok") ++completed; else if(f.status=="failed") ++failed; else ++cancelled;}
    s << "{\"schema_version\":1,\"status\":" << json_quote(run.error.empty()?"ok":"failed")
      << ",\"error\":" << (run.error.empty()?"null":json_quote(run.error))
      << ",\"sdk_enabled\":" << (sdk_available()?"true":"false") << ",\"model_path\":" << json_quote(o.model)
      << ",\"input\":{\"mode\":" << json_quote(!o.video.empty()?"video_file":(!o.image_list.empty()?"image_list":"image"))
      << ",\"image\":" << json_quote(o.image) << ",\"image_list\":" << json_quote(o.image_list)
      << ",\"video\":" << json_quote(o.video) << ",\"relative_list_paths\":\"relative to list parent directory\""
      << ",\"iterations\":" << o.iterations << ",\"iteration_semantics\":"
      << json_quote(o.video.empty()?"complete passes over images":"maximum frames; nonempty EOF ends run")
      << ",\"video_eof\":" << (run.video_eof?"true":"false") << "}"
      << ",\"preprocessing\":{\"geometry\":" << json_quote(o.geometry)
      << ",\"interpolation\":\"OpenCV INTER_LINEAR\",\"color\":\"RGB\",\"submitted_type\":\"UINT8\""
      << ",\"submitted_layout\":\"contiguous NHWC batch1\",\"normalization\":\"none in application; SDK conversion pass_through=0\""
      << ",\"letterbox_fill\":114,\"rounding\":\"floor(x+0.5)\"}"
      << ",\"postprocessing\":{\"confidence\":0.25,\"nms_iou\":0.45,\"max_detections\":100,\"classes\":80"
      << ",\"candidate_count\":8400,\"nms\":\"CPU class-aware in network coordinates before inverse mapping and clipping\""
      << ",\"score_tie_policy\":\"ascending class then descending anchor; NumPy unstable sort exact ties may differ\""
      << ",\"clip\":\"[0,width-1] and [0,height-1]\",\"bbox_rounding\":\"nearest-even\"}"
      << ",\"timing_boundaries\":{\"clock\":\"steady_clock\",\"decode_ms\":\"file imread / video read; measured wall includes decoding\""
      << ",\"producer_wait_ms\":\"push entry to accepted enqueue, includes bounded-queue backpressure\""
      << ",\"queue_ms\":\"accepted enqueue to worker dequeue; excludes producer wait\""
      << ",\"preprocess_ms\":\"geometry, resize/pad, BGR-to-RGB, contiguous copy\""
      << ",\"inference_call_ms\":\"inputs_set plus synchronous rknn_run; not isolated NPU latency\""
      << ",\"output_and_nms_ms\":\"float outputs_get/copy/release plus validation/decode/CPU NMS\""
      << ",\"total_wall_ms\":\"same-frame decode entry to completed postprocessing; directly timed, not sum of P50s\""
      << ",\"measurement_wall_ms\":\"thread launch through producer finish and worker joins; excludes initialization/startup decode/warmup/report write\""
      << ",\"percentiles\":\"nearest rank, completed frames only, first measured frame included\""
      << ",\"initialization_ms\":\"per worker: model file read, init, metadata queries, core-mask setup\""
      << ",\"startup_decode_ms\":\"initial sample decode and video open/close, excluded from measurement\""
      << ",\"warmup_wall_ms\":\"per worker full preprocess/infer/postprocess loop on the startup sample; no file decode\""
      << ",\"first_call_wall_ms\":\"per worker first preprocess through postprocess call, excluding decode and queue; phase is explicitly listed\"}"
      << ",\"unsupported\":[\"camera\",\"DMABUF\",\"mixed quantization\",\"hardware-only profiling\",\"power measurement\",\"frequency controls\"]"
      << ",\"startup_decode_ms\":" << json_number(run.startup_decode_ms) << ",\"workers\":[";
    for(std::size_t i=0;i<run.workers.size();++i) {
        const auto& w=run.workers[i]; if(i) s << ',';
        s << "{\"worker\":" << i << ",\"requested_core_mask\":" << (1<<i)
          << ",\"context_initialized\":" << (w.runtime?"true":"false")
          << ",\"initialization_ms\":" << json_number(w.initialization_ms)
          << ",\"first_call_wall_ms\":" << json_number(w.first_call_wall_ms)
          << ",\"first_call_phase\":" << json_quote(o.warmup>0?"warmup":"measurement")
          << ",\"warmup_requested\":" << o.warmup << ",\"warmup_completed\":" << w.warmup_completed
          << ",\"warmup_wall_ms\":" << json_number(w.warmup_wall_ms)
          << ",\"error\":" << (w.error.empty()?"null":json_quote(w.error))
          << ",\"metadata\":";
        if(!w.runtime) s << "null";
        else {
            const auto& m=w.metadata;
            s << "{\"provider\":" << json_quote(m.provider) << ",\"core_mask\":" << m.core_mask
              << ",\"sdk_version\":" << json_quote(m.sdk_version) << ",\"driver_version\":" << json_quote(m.driver_version)
              << ",\"output_layout\":" << json_quote(m.layout) << ",\"input\":" << tensor_json(m.input) << ",\"outputs\":[";
            for(std::size_t j=0;j<m.outputs.size();++j) {if(j) s << ','; s << tensor_json(m.outputs[j]);}
            s << "]}";
        }
        s << '}';
    }
    s << "],\"first_measured_frame\":";
    if(run.frames.empty()) s << "null";
    else {
        const auto& f=run.frames.front();
        s << "{\"frame_id\":" << f.id << ",\"status\":" << json_quote(f.status)
          << ",\"worker\":" << f.worker << ",\"total_wall_ms\":" << json_number(f.total_wall_ms)
          << ",\"preprocess_ms\":" << json_number(f.preprocess_ms)
          << ",\"inference_call_ms\":" << json_number(f.inference_call_ms)
          << ",\"output_and_nms_ms\":" << json_number(f.output_and_nms_ms) << '}';
    }
    const auto& q=run.queue;
    s << ",\"queue\":{\"capacity\":" << o.queue_capacity << ",\"policy\":\"wait; no drop except explicit fatal/signal cancellation\""
      << ",\"peak\":" << q.peak << ",\"pushes\":" << q.pushes << ",\"pops\":" << q.pops
      << ",\"normal_drops\":0,\"active_sdk_call_cancellation\":\"SDK has no per-call timeout here; termination waits for an in-flight synchronous driver call to return\""
      << ",\"cancelled_entries\":" << q.cancelled_entries
      << ",\"cancelled\":" << (q.cancelled?"true":"false")
      << ",\"producer_wait_total_ms\":" << json_number(q.producer_wait_ms)
      << ",\"consumer_wait_total_ms\":" << json_number(q.consumer_wait_ms) << '}'
      << ",\"completed\":" << completed << ",\"failed_frames\":" << failed << ",\"cancelled_frames\":" << cancelled
      << ",\"measurement_wall_ms\":" << json_number(run.measurement_wall_ms)
      << ",\"throughput_completed_fps\":" << json_number(run.measurement_wall_ms>0?completed*1000.0/run.measurement_wall_ms:missing)
      << ",\"summary_ms\":{";
    const std::pair<const char*,double FrameResult::*> metrics[]={
        {"decode_ms",&FrameResult::decode_ms},{"producer_wait_ms",&FrameResult::producer_wait_ms},
        {"queue_ms",&FrameResult::queue_ms},{"preprocess_ms",&FrameResult::preprocess_ms},
        {"inference_call_ms",&FrameResult::inference_call_ms},{"output_and_nms_ms",&FrameResult::output_and_nms_ms},
        {"total_wall_ms",&FrameResult::total_wall_ms}};
    bool first_metric=true;
    for(const auto& metric:metrics) {
        std::vector<double> values;
        for(const auto& frame:run.frames) if(frame.status=="ok") values.push_back(frame.*metric.second);
        if(!first_metric) s << ',';
        first_metric=false;
        s << json_quote(metric.first) << ":{\"p50\":" << percentile(values,.50) << ",\"p95\":" << percentile(values,.95) << '}';
    }
    s << "},\"frames\":[";
    for(std::size_t i=0;i<run.frames.size();++i) {
        const auto& f=run.frames[i]; if(i) s << ',';
        s << "{\"frame_id\":" << f.id << ",\"source\":" << json_quote(f.source) << ",\"worker\":" << f.worker
          << ",\"status\":" << json_quote(f.status) << ",\"error\":" << (f.error.empty()?"null":json_quote(f.error))
          << ",\"timings_ms\":{";
        bool first=true;
        for(const auto& metric:metrics) {if(!first) s << ','; first=false; s << json_quote(metric.first) << ':' << json_number(f.*metric.second);}
        s << "},\"transform\":";
        if(f.transform.mode.empty()) s << "null"; else s << transform_json(f.transform);
        s << ",\"detections\":[";
        for(std::size_t j=0;j<f.detections.size();++j) {
            const auto& d=f.detections[j]; if(j) s << ',';
            s << "{\"class_id\":" << d.class_id << ",\"class_name\":" << json_quote(d.class_name)
              << ",\"score\":" << json_number(d.score) << ",\"bbox\":[";
            for(int k=0;k<4;++k) {if(k) s << ','; s << d.bbox[k];}
            s << "],\"source_box\":[";
            for(int k=0;k<4;++k) {if(k) s << ','; s << json_number(d.source_box[k]);}
            s << "],\"center\":[" << d.center[0] << ',' << d.center[1] << "]}";
        }
        s << "]}";
    }
    s << "]}\n"; return s.str();
}
#if EDGEAI_HAVE_OPENCV
struct Task {
    FrameResult frame;
    cv::Mat image;
    Clock::time_point decode_started;
};
void pipeline(const Options& o,const std::vector<std::string>& paths,Run& run) {
    if(!sdk_available()) throw std::runtime_error("RKNN SDK/runtime support is disabled; no mock inference");
    cv::Mat sample;
    cv::VideoCapture video;
    const auto startup=Clock::now();
    if(!o.video.empty()) {
        if(!fs::is_regular_file(o.video)) throw std::runtime_error("--video requires a regular file; devices/URLs are forbidden");
        if(!video.open(o.video,cv::CAP_ANY) || !video.read(sample) || sample.empty())
            throw std::runtime_error("Video file contains no decodable frames");
        video.release(); // measured video starts at original frame 0, not frame 1
    } else {
        if(paths.empty() || !fs::is_regular_file(paths.front())) throw std::runtime_error("Image file is missing or not regular");
        sample=cv::imread(paths.front(),cv::IMREAD_COLOR);
        if(sample.empty()) throw std::runtime_error("Image cannot be decoded");
    }
    run.startup_decode_ms=milliseconds(startup);
    run.workers.resize(static_cast<std::size_t>(o.workers));
    for(int i=0;i<o.workers;++i) {
        check_stop(); auto& w=run.workers[i]; const auto begin=Clock::now();
        try {w.runtime=create_runtime(o.model,1<<i); w.metadata=w.runtime->info();}
        catch(const std::exception& e) {w.initialization_ms=milliseconds(begin);w.error=e.what();throw;}
        w.initialization_ms=milliseconds(begin);
        const auto warmup_started=Clock::now();
        try {
            for(int n=0;n<o.warmup;++n) {
                check_stop(); const auto first=Clock::now();
                auto input=preprocess(sample,w.metadata.input_width,w.metadata.input_height,o.geometry);
                auto inference=w.runtime->infer(input); (void)decode(inference.outputs,input.transform);
                if(n==0) w.first_call_wall_ms=milliseconds(first);
                ++w.warmup_completed;
            }
        } catch(const std::exception& e) {w.warmup_wall_ms=milliseconds(warmup_started);w.error=e.what();throw;}
        w.warmup_wall_ms=milliseconds(warmup_started);
    }
    BoundedQueue<Task> queue(static_cast<std::size_t>(o.queue_capacity),[]{return interrupted!=0;});
    std::vector<std::thread> threads;
    const auto measured_started=Clock::now();
    try {
        for(int i=0;i<o.workers;++i) threads.emplace_back([&,i] {
            auto& w=run.workers[i];
            try {
                while(auto entry=queue.pop()) {
                    auto& task=entry->value; auto& f=task.frame;
                    f.worker=i; f.queue_ms=milliseconds(entry->enqueued); f.producer_wait_ms=entry->producer_wait_ms;
                    const auto first=Clock::now();
                    try {
                        const auto prep=Clock::now();
                        auto input=preprocess(task.image,w.metadata.input_width,w.metadata.input_height,o.geometry);
                        f.preprocess_ms=milliseconds(prep); f.transform=input.transform;
                        auto inference=w.runtime->infer(input);
                        f.inference_call_ms=inference.inference_call_ms;
                        const auto post=Clock::now(); f.detections=decode(inference.outputs,input.transform);
                        f.output_and_nms_ms=inference.output_ms+milliseconds(post);
                        f.total_wall_ms=milliseconds(task.decode_started); f.status="ok";
                        if(!std::isfinite(w.first_call_wall_ms)) w.first_call_wall_ms=milliseconds(first);
                    } catch(const std::exception& e) {
                        f.total_wall_ms=milliseconds(task.decode_started);f.status="failed";f.error=e.what();
                        w.error="Frame "+std::to_string(f.id)+": "+e.what(); queue.cancel();
                    }
                    w.results.push_back(std::move(f));
                    if(!w.error.empty()) break;
                }
            } catch(const std::exception& e) {w.error=std::string("Worker failure: ")+e.what();queue.cancel();}
            catch(...) {w.error="Unknown worker failure";queue.cancel();}
        });
        if(!o.video.empty() && !video.open(o.video,cv::CAP_ANY)) throw std::runtime_error("Cannot reopen measured video input");
        const std::size_t count=o.video.empty()?paths.size()*o.iterations:static_cast<std::size_t>(o.iterations);
        for(std::size_t id=0;id<count;++id) {
            check_stop(); if(queue.cancelled()) break;
            Task task; task.frame.id=id;
            task.frame.source=o.video.empty()?paths[id%paths.size()]:fs::absolute(o.video).string()+"#frame="+std::to_string(id);
            task.decode_started=Clock::now();
            try {
                if(o.video.empty()) {
                    if(!fs::is_regular_file(task.frame.source)) throw std::runtime_error("Image missing/not a regular file");
                    task.image=cv::imread(task.frame.source,cv::IMREAD_COLOR);
                } else if(!video.read(task.image)) {
                    run.video_eof=true;
                    if(id==0) throw std::runtime_error("Measured video input has no frames");
                    break;
                }
                if(task.image.empty()) throw std::runtime_error("Input frame cannot be decoded");
                task.frame.decode_ms=milliseconds(task.decode_started);
            } catch(const std::exception& e) {
                task.frame.status="failed";task.frame.error=e.what();task.frame.decode_ms=milliseconds(task.decode_started);
                task.frame.total_wall_ms=task.frame.decode_ms;run.attempts.push_back(task.frame);throw;
            }
            run.attempts.push_back(task.frame);
            if(!queue.push(std::move(task))) break;
        }
        queue.close();
    } catch(const std::exception& e) {run.error=std::string("Producer/pipeline failure: ")+e.what();queue.cancel();}
    catch(...) {run.error="Unknown producer/pipeline failure";queue.cancel();}
    // Always join after cancellation, including partial thread creation failure.
    for(auto& thread:threads) if(thread.joinable()) thread.join();
    run.measurement_wall_ms=milliseconds(measured_started); run.queue=queue.stats();
    std::map<std::uint64_t,FrameResult> results;
    for(auto& w:run.workers) {
        if(!w.error.empty() && run.error.empty()) run.error=w.error;
        for(auto& f:w.results) results.emplace(f.id,std::move(f));
    }
    for(auto& attempt:run.attempts) {
        auto found=results.find(attempt.id);
        if(found!=results.end()) run.frames.push_back(std::move(found->second));
        else {
            if(attempt.status!="failed") attempt.error="Cancelled before completion (fatal error or signal)";
            run.frames.push_back(std::move(attempt));
        }
    }
    if(interrupted && run.error.empty()) run.error="Interrupted by signal";
    if(run.queue.cancelled && run.error.empty()) run.error="Pipeline cancelled";
    if(run.frames.empty() && run.error.empty()) run.error="No input frames completed";
}
#endif
} // namespace

int main(int argc,char** argv) {
    std::signal(SIGINT,signal_handler); std::signal(SIGTERM,signal_handler);
    try {
        std::vector<std::string> args; for(int i=1;i<argc;++i) args.emplace_back(argv[i]);
        const auto options=edgeai::yolo::parse_options(args);
        if(options.help) {std::cout << edgeai::yolo::usage();return 0;}
        auto paths=image_paths(options); no_report_alias(options,paths);
        ReportFile report(options.report,options.overwrite);
        Run run;
        try {
#if EDGEAI_HAVE_OPENCV
            pipeline(options,paths,run);
#else
            throw std::runtime_error("OpenCV support is disabled; real vision unavailable (no mock)");
#endif
        } catch(const std::exception& e) {run.error=e.what();}
        report.write(render(options,run));
        if(!run.error.empty()) {std::cerr << "edge_yolo_vision failed: " << run.error << '\n';return 1;}
        std::cout << "edge_yolo_vision completed " << run.frames.size() << " frames; report: " << options.report << '\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "edge_yolo_vision: " << e.what() << '\n' << edgeai::yolo::usage();return 2;
    }
}
