#include "yolo_vision.hpp"
#include <atomic>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#ifndef EDGEAI_HAVE_OPENCV
#define EDGEAI_HAVE_OPENCV 1
#endif
#if EDGEAI_HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {
using namespace edgeai::yolo;
int checks=0;
void check(bool condition,const char* description) {
    ++checks; if(!condition) throw std::runtime_error(description);
}
template<class F> void rejects(F f,const char* description) {
    bool rejected=false;
    try {f();} catch(const std::exception&) {rejected=true;}
    check(rejected,description);
}
void near(double a,double b,const char* description) {check(std::abs(a-b)<1e-9,description);}
Tensor prediction() {return {{1,84,8400},std::vector<float>(84*8400,0)};}
void candidate(Tensor& p,std::size_t anchor,int cls,float score,float cx=320,float cy=320,float w=128,float h=128) {
    p.values[anchor]=cx; p.values[8400+anchor]=cy; p.values[16800+anchor]=w;p.values[25200+anchor]=h;
    p.values[(4+cls)*8400+anchor]=score;
}
void geometry_tests() {
    const auto resize=make_transform(1280,720,640,640,"resize");
    near(resize.scale_x,.5,"resize scale x");near(resize.scale_y,640.0/720,"resize scale y");
    check(resize.pad_left==0 && resize.pad_top==0,"resize has no padding");
    const auto odd=make_transform(5,3,8,8,"letterbox");
    check(odd.resized_width==8 && odd.resized_height==5 && odd.pad_top==1,"odd padding puts remainder bottom");
    check(odd.target_height-odd.resized_height-odd.pad_top==2,"odd bottom padding");
    near(odd.scale_y,5.0/3,"actual rounded scale not nominal scale");
    const auto half=make_transform(4,1,6,6,"letterbox");
    check(half.resized_height==2,"half-up exact 1.5 becomes 2");
    rejects([]{make_transform(0,1,640,640,"resize");},"zero source rejected");
    rejects([]{make_transform(1,1,0,640,"resize");},"zero target rejected");
    rejects([]{make_transform(1,1,640,640,"crop");},"unknown geometry rejected");
    rejects([]{make_transform(10000,1,1,1,"letterbox");},"zero rounded height rejected");
    auto bad=odd;bad.scale_y=1;rejects([&]{validate_transform(bad);},"incorrect actual scale rejected");
    bad=odd;bad.pad_top=2;rejects([&]{validate_transform(bad);},"incorrect pad rejected");
    bad=odd;bad.scale_x=std::numeric_limits<double>::quiet_NaN();
    rejects([&]{validate_transform(bad);},"NaN geometry rejected");
    const auto json=transform_json(odd);
    check(json.find("\"resized_height\":5")!=std::string::npos,"transform JSON actual resize");
    check(json.find("pad_right")==std::string::npos,"transform JSON exact supported field set");
}
void decode_tests() {
    auto p=prediction();candidate(p,0,0,.9f);
    const auto t=make_transform(1280,720,640,640,"resize");
    auto result=decode({p},t);
    check(result.size()==1 && result[0].class_name=="person","single COCO head decoded");
    near(result[0].source_box[0],512,"resize inverse x1");near(result[0].source_box[1],288,"resize inverse y1");
    Tensor coordinates{{1,4,8400},std::vector<float>(p.values.begin(),p.values.begin()+4*8400)};
    Tensor scores{{1,80,8400},std::vector<float>(p.values.begin()+4*8400,p.values.end())};
    const auto split=decode({coordinates,scores},t),reversed=decode({scores,coordinates},t);
    check(split.size()==result.size() && split[0].bbox==result[0].bbox && split[0].score==result[0].score,
          "split and single outputs equivalent");
    check(reversed[0].bbox==result[0].bbox,"split head order autodetected from shape");
    result=decode({p},make_transform(1280,720,640,640,"letterbox"));
    near(result[0].source_box[1],232,"letterbox inverse removes top padding");
    near(result[0].source_box[3],488,"letterbox inverse bottom coordinate");
    p=prediction();candidate(p,0,0,.9f,320,320,1000,1000);
    result=decode({p},make_transform(640,640,640,640,"resize"));
    check(result[0].bbox==std::array<int,4>{0,0,639,639},"clip to Python pixel limits");
    p=prediction();candidate(p,0,0,.9f,2,2,3,3);
    result=decode({p},make_transform(640,640,640,640,"resize"));
    check(result[0].bbox==std::array<int,4>{0,0,4,4},"bbox ties round nearest-even");
    p=prediction();candidate(p,0,0,.9f);candidate(p,1,0,.8f);candidate(p,2,1,.85f);
    result=decode({p},make_transform(640,640,640,640,"resize"));
    check(result.size()==2 && result[0].class_id==0 && result[1].class_id==1,"NMS suppresses same class, preserves different classes");
    check(decode({p},t,.95).empty(),"confidence filter");
    check(decode({p},t,.25,.45,1).size()==1,"max detections limit");
    p=prediction();candidate(p,0,0,.9f,-250,20,700,20);candidate(p,1,0,.8f,-55,20,290,20);
    result=decode({p},make_transform(640,640,640,640,"resize"));
    check(result.size()==2,"NMS is before clip: clipped IoU must not suppress network-distinct boxes");
    p=prediction();candidate(p,0,0,.9f,4,3,4,2);
    result=decode({p},make_transform(5,3,8,8,"letterbox"));
    near(result[0].source_box[0],1.25,"odd letterbox inverse x uses actual rounded scale");
    near(result[0].source_box[1],.6,"odd letterbox inverse y uses actual rounded scale");
    near(result[0].source_box[3],1.8,"odd letterbox bottom inverse");
    p=prediction();candidate(p,0,0,.9f,-100,-100,1,1);
    check(decode({p},t).empty(),"clipped degenerate boxes omitted");
    p=prediction();candidate(p,0,0,.25f);
    check(decode({p},t).size()==1,"confidence threshold inclusive");
    p.values.back()=std::numeric_limits<float>::infinity();
    rejects([&]{decode({p},t);},"Inf in unused low confidence anchor rejects entire frame");
    p.values.back()=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{decode({p},t);},"NaN rejects entire frame");
    p=prediction();candidate(p,0,0,.9f,std::numeric_limits<float>::max(),320,std::numeric_limits<float>::max(),128);
    rejects([&]{decode({p},t);},"finite output with overflowing float32 xyxy conversion rejected");
    p=prediction();p.values.pop_back();rejects([&]{decode({p},t);},"short output rejected");
    rejects([]{validate_output_layout({{1,8400,84}});},"transposed undocumented layout rejected");
    rejects([]{validate_output_layout({{2,84,8400}});},"batch2 rejected");
    rejects([]{validate_output_layout({{1,85,8400}});},"wrong class count rejected");
    rejects([]{validate_output_layout({{1,4,8400},{1,80,8399}});},"anchor mismatch rejected");
    rejects([]{validate_output_layout({});},"no output rejected");
    p=prediction();rejects([&]{decode({p},t,std::numeric_limits<double>::quiet_NaN());},"NaN threshold rejected");
    rejects([&]{decode({p},t,.25,1.1);},"invalid NMS threshold rejected");
    rejects([&]{decode({p},t,.25,.45,0);},"zero result limit rejected");
}
void option_and_json_tests() {
    const std::vector<std::string> base={"--model","missing.rknn","--image","input.png","--report","out.json"};
    auto o=parse_options(base);check(o.workers==1 && o.geometry=="resize" && o.warmup==0,"CLI defaults");
    auto args=base;args.insert(args.end(),{"--workers","3","--geometry","letterbox","--warmup","2","--overwrite"});
    o=parse_options(args);check(o.workers==3 && o.warmup==2 && o.overwrite,"CLI valid options");
    for(const auto& pair:std::vector<std::pair<std::string,std::string>>{
            {"--workers","2"},{"--workers","0"},{"--iterations","0"},{"--iterations","NaN"},
            {"--iterations","1.5"},{"--iterations","-1"},{"--iterations","1000001"},
            {"--warmup","inf"},{"--queue-capacity","0"},{"--queue-capacity","1025"},{"--geometry","crop"}}) {
        args=base;args.push_back(pair.first);args.push_back(pair.second);
        rejects([&]{parse_options(args);},"CLI bad value rejected");
    }
    args=base;args.insert(args.end(),{"--video","clip.mp4"});rejects([&]{parse_options(args);},"ambiguous input mode rejected");
    args=base;args.insert(args.end(),{"--model","other.rknn"});rejects([&]{parse_options(args);},"duplicate option rejected");
    args=base;args.push_back("--warmup");rejects([&]{parse_options(args);},"missing argument rejected");
    args=base;args.insert(args.end(),{"--mock","true"});rejects([&]{parse_options(args);},"mock option unsupported");
    rejects([]{parse_options({});},"missing required arguments rejected");
    check(parse_options({"--help"}).help,"help without resources");
    check(json_quote(std::string("a\n\"\\\t"))=="\"a\\u000a\\\"\\\\\\u0009\"","JSON control characters escaped");
    check(json_number(std::numeric_limits<double>::infinity())=="null","JSON Inf becomes null");
    check(json_number(std::numeric_limits<double>::quiet_NaN())=="null","JSON NaN becomes null");
    if(!sdk_available()) rejects([]{create_runtime("missing.rknn",1);},"SDK-off cannot silently run inference");
    else rejects([]{create_runtime("missing.rknn",8);},"invalid core mask fails before runtime access");
}
void affine_metadata_tests() {
    for(const std::string type:{"FP16","FP32"}) {
        validate_affine_metadata(type,1.0,0);check(true,"verified floating identity affine accepted");
        rejects([&]{validate_affine_metadata(type,.5,0);},"floating nonidentity affine scale rejected");
        rejects([&]{validate_affine_metadata(type,std::nextafter(1.0,2.0),0);},"floating affine requires exact scale one");
        rejects([&]{validate_affine_metadata(type,1.0,1);},"floating affine nonzero zero point rejected");
        rejects([&]{validate_affine_metadata(type,1.0,-1);},"floating affine negative zero point rejected");
    }
    for(int zp:{-128,0,127}) {validate_affine_metadata("INT8",.01,zp);check(true,"INT8 affine range endpoints accepted");}
    for(int zp:{0,128,255}) {validate_affine_metadata("UINT8",.01,zp);check(true,"UINT8 affine range endpoints accepted");}
    rejects([]{validate_affine_metadata("INT8",1,-129);},"INT8 low out-of-range zero point rejected");
    rejects([]{validate_affine_metadata("INT8",1,128);},"INT8 high out-of-range zero point rejected");
    rejects([]{validate_affine_metadata("UINT8",1,-1);},"UINT8 negative zero point rejected");
    rejects([]{validate_affine_metadata("UINT8",1,256);},"UINT8 high out-of-range zero point rejected");
    for(const std::string type:{"FP16","FP32","INT8","UINT8"}) {
        for(double scale:{0.0,-1.0,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
            rejects([&]{validate_affine_metadata(type,scale,0);},"all affine types reject nonpositive/nonfinite scale");
    }
    rejects([]{validate_affine_metadata("FP64",1,0);},"unknown affine tensor type rejected");
}
void queue_tests() {
    rejects([]{BoundedQueue<int> q(0);},"zero queue capacity rejected");
    BoundedQueue<int> q(2);check(q.push(10) && q.push(20),"bounded queue accepts capacity");q.close();
    check(q.pop()->value==10 && q.pop()->value==20 && !q.pop(),"close drains in FIFO order and wakes consumers");
    check(!q.push(30),"closed queue refuses producer");
    check(q.stats().peak==2 && q.stats().pushes==2 && q.stats().pops==2,"queue observable counters");
    BoundedQueue<int> blocked(1);blocked.push(1);
    std::promise<void> started;
    auto pending=std::async(std::launch::async,[&]{started.set_value();return blocked.push(2);});
    started.get_future().wait();blocked.cancel();
    check(pending.wait_for(std::chrono::seconds(2))==std::future_status::ready && !pending.get(),"worker failure cancellation wakes blocked producer");
    check(blocked.stats().cancelled_entries==1 && !blocked.pop(),"cancel discards pending work");
    BoundedQueue<int> empty(1);std::promise<void> consumer_started;
    auto consumer=std::async(std::launch::async,[&]{consumer_started.set_value();return empty.pop();});
    consumer_started.get_future().wait();empty.cancel();
    check(consumer.wait_for(std::chrono::seconds(2))==std::future_status::ready && !consumer.get(),"producer failure wakes blocked consumer");
    std::atomic<bool> stop{false};BoundedQueue<int> signal_queue(1,[&]{return stop.load();});signal_queue.push(1);
    auto signal_wait=std::async(std::launch::async,[&]{return signal_queue.push(2);});stop=true;
    check(signal_wait.wait_for(std::chrono::seconds(2))==std::future_status::ready && !signal_wait.get(),"signal predicate releases blocked queue");
    BoundedQueue<int> concurrent(3);std::atomic<int> count{0},sum{0};std::vector<std::thread> threads;
    for(int i=0;i<3;++i) threads.emplace_back([&]{while(auto e=concurrent.pop()) {++count;sum+=e->value;}});
    for(int i=0;i<50;++i) check(concurrent.push(i),"three-worker enqueue");
    concurrent.close();for(auto& thread:threads) thread.join();
    check(count==50 && sum==1225 && concurrent.stats().peak<=3,"three-worker FIFO queue processes each ID once");
}
void color_tests() {
#if EDGEAI_HAVE_OPENCV
    cv::Mat bgr(1,2,CV_8UC3);bgr.at<cv::Vec3b>(0,0)={10,20,30};bgr.at<cv::Vec3b>(0,1)={40,50,60};
    const auto rgb=preprocess(bgr,2,1,"resize");
    check(rgb.rgb==std::vector<std::uint8_t>{30,20,10,60,50,40},"BGR converts to RGB uint8 NHWC without normalization");
    cv::Mat solid(3,5,CV_8UC3,cv::Scalar(10,20,30));const auto padded=preprocess(solid,8,8,"letterbox");
    check(padded.rgb.size()==8*8*3,"letterbox dense storage");
    check(padded.rgb[0]==114 && padded.rgb[8*3]==30 && padded.rgb[6*8*3]==114,"odd top/bottom pad RGB fill114");
    cv::Mat storage(4,6,CV_8UC3,cv::Scalar(1,2,3));cv::Mat roi=storage(cv::Rect(1,1,3,2));
    check(!roi.isContinuous() && preprocess(roi,3,2,"resize").rgb.size()==18,"noncontiguous source becomes contiguous RGB");
    rejects([]{cv::Mat empty;preprocess(empty,640,640,"resize");},"empty preprocessing fails");
    rejects([]{cv::Mat gray(2,2,CV_8UC1);preprocess(gray,640,640,"resize");},"non-BGR source fails");
#else
    std::cout << "SKIP: OpenCV color/INTER_LINEAR/ROI tests (development SDK unavailable)\n";
#endif
}
} // namespace
int main() {
    try {
        geometry_tests();decode_tests();option_and_json_tests();affine_metadata_tests();queue_tests();color_tests();
        std::cout << "PASS: " << checks << " offline checks; RKNN inference is not exercised\n";return 0;
    } catch(const std::exception& e) {std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';return 1;}
}
