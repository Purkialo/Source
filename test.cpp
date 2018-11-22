#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <utility>
#include <chrono>
#include <cmath>

#include <ie_cnn_net_reader.h>
#include <ie_plugin_dispatcher.hpp>
#include <ie_infer_request.hpp>
#include "include/format_reader_ptr.h"
#include "include/common.hpp"

using namespace InferenceEngine;

int pri_image_size = 400;
int shrinkage[6] = {16,32,64,100,150,300};
int feature_map_size[6] = {25,13,7,4,2,1};
int box_sizes_min[6] = {60,105,150,195,240,285};
int box_sizes_max[6] = {105,150,195,240,285,330};
float center_variance = 0.1;
float size_variance = 0.2;
const int length = 5184;
float nms_prob_threshold = 0.2;
int nms_candidate_size = 200;
int nms_topk = 10;
float nms_iou_threshold = 0.45;
float nms_sigma = 0.5;

int main(int argc, char *argv[]){
    void genprior(float priors[][4]);
    void convert_locations_to_boxes(float boxes[][4],const float priors[][4],const float * detection);

    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;
    //读取模型
    CNNNetReader network_reader;
    network_reader.ReadNetwork("../resssd.xml");
    network_reader.ReadWeights("../resssd.bin");
    CNNNetwork network = network_reader.getNetwork();
    //加载输入格式
    InputsDataMap input_info(network.getInputsInfo());
    //加载输出格式
    OutputsDataMap output_info(network.getOutputsInfo());
    //加载插件
    std::cout << "Loading plugin" << std::endl;
    InferencePlugin plugin = PluginDispatcher({"/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/"}).getPluginByDevice("CPU");
    //处理输入
    std::cout << "Preparing input blobs" << std::endl;
    std::string imageInputName;
    InputInfo::Ptr inputInfo = input_info.begin()->second;

    SizeVector inputImageDims;

    for (auto & item : input_info) {
        /** Working with first input tensor that stores image **/
        if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
            imageInputName = item.first;

            std::cout << "Batch size is " << std::to_string(network_reader.getNetwork().getBatchSize()) << std::endl;

            /** Creating first input blob **/
            Precision inputPrecision = Precision::U8;
            item.second->setPrecision(inputPrecision);
        }
    }
    //处理输出
    std::cout << "Preparing output blobs" << std::endl;

    std::string outputName;
    DataPtr outputInfo;
    for (const auto& out : output_info) {
        outputName = out.first;
        outputInfo = out.second;
    }
    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();
    outputInfo->setPrecision(Precision::FP32);
    std::cout << "Output layer is " << outputName << std::endl;
    std::cout << "Batch size is " << outputDims[0] << std::endl;
    std::cout << "Batch size is " << outputDims[1] << std::endl;
    std::cout << "Batch size is " << outputDims[2] << std::endl;
    std::cout << "Batch size is " << outputDims[3] << std::endl;
    const int maxProposalCount = outputDims[2];
    const int objectSize = outputDims[3];

    //加载模型
    std::cout << "Loading model to the plugin" << std::endl;
    ExecutableNetwork my_network = plugin.LoadNetwork(network, {});
    //创建INFER请求
    InferRequest infer_request = my_network.CreateInferRequest();

    // --------------------------- 9. Prepare input --------------------------------------------------------
    /** Collect images data ptrs **/
    std::vector<std::string> images;
    images.push_back("/home/student/cxg_workspace/test-ov/1");
    images.push_back("/home/student/cxg_workspace/test-ov/1");

    std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
    std::vector<int> imageWidths, imageHeights;

    for (auto & i : images) {
        FormatReader::ReaderPtr reader(i.c_str());
        if (reader.get() == nullptr) {
            std::cout << "Image " + i + " cannot be read!" << std::endl;
        }
        /** Store image data **/
        std::shared_ptr<unsigned char> originalData(reader->getData());
        std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
        if (data.get() != nullptr) {
            originalImagesData.push_back(originalData);
            imagesData.push_back(data);
            imageWidths.push_back(reader->width());
            imageHeights.push_back(reader->height());
        }
    }
    if (imagesData.empty())
        std::cout << "Valid input images were not found!" << std::endl;

    size_t batchSize = network.getBatchSize();
    std::cout << "Batch size is " << std::to_string(batchSize) << std::endl;
    if (batchSize != imagesData.size()) {
        std::cout << "Number of images " + std::to_string(imagesData.size()) + \
            " doesn't match batch size " + std::to_string(batchSize) << std::endl;
        std::cout << std::to_string(std::min(imagesData.size(), batchSize)) + \
            " images will be processed" << std::endl;
        batchSize = std::min(batchSize, imagesData.size());
    }

    /** Creating input blob **/
    Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

    /** Filling input tensor with images. First b channel, then g and r channels **/
    size_t num_channels = imageInput->getTensorDesc().getDims()[1];
    size_t image_size = imageInput->getTensorDesc().getDims()[3] * imageInput->getTensorDesc().getDims()[2];

    unsigned char* my_data = static_cast<unsigned char*>(imageInput->buffer());
    //auto my_data = imageInput->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    /** Iterate over all input images **/
    for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** Iterate over all channels **/
            for (size_t ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes            **/
                if(ch == 0)
                    my_data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + 2];
                if(ch == 1)
                    my_data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + 1];
                if(ch == 2)
                    my_data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + 0];
            }
        }
    }
    // for(int myk = 0; myk < 10; myk++){
    //     std::cout << (int)my_data[myk] <<' ';// << std::endl;
    // }
    //     std::cout << std::endl;
    // for(int myk = image_size; myk < image_size + 10; myk++){
    //     std::cout << (int)my_data[myk] <<' ';// << std::endl;
    // }
    //     std::cout << std::endl;
    // for(int myk = image_size * 2; myk < image_size * 2 + 10; myk++){
    //     std::cout << (int)my_data[myk] <<' ';// << std::endl;
    // }
    //     std::cout << std::endl;

    std::cout << "Start inference (" << 1 << " iterations)" << std::endl;

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    double total = 0.0;
    /** Start inference & calc performance **/
    for (int iter = 0; iter < 1; ++iter) {
        auto t0 = Time::now();
        infer_request.Infer();
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total += d.count();
    }

    std::cout << "Processing output blobs" << std::endl;

    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    // Const
    float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    float priors[length][4];
    float boxes[length][4];
    genprior(priors);
    convert_locations_to_boxes(boxes,priors,(const float *)detection);
    std::vector<float> subvector;
    std::vector<std::vector<float>> boxes_probs;

    for(int myk = 0; myk < length; myk++){
        subvector.push_back(boxes[myk][0]);
        subvector.push_back(boxes[myk][1]);
        subvector.push_back(boxes[myk][2]);
        subvector.push_back(boxes[myk][3]);
        subvector.push_back(detection[myk * 6]);
        boxes_probs.push_back(subvector);
        subvector.clear();
    }
    std::vector<std::vector<float>> subset_boxes_probs;
    for(int i; i < boxes_probs.size(); i++){
        if(boxes_probs[i][4] > nms_prob_threshold){
            subset_boxes_probs.push_back(boxes_probs[i]);
        }
    }

    // for(int myk = 0; myk < 60; myk+=6){
    //     float exp_1 = exp(detection[myk]);
    //     float exp_2 = exp(detection[myk + 1]);
    //     detection[myk] = exp_1/(exp_1 + exp_2);
    //     detection[myk + 1] = exp_2/(exp_1 + exp_2);
    //     std::cout << myk / 6 + 1 << ' ' << detection[myk] << ' ' << detection[myk + 1]  << ' ';
    //     std::cout << boxes[myk / 6][0] << ' ' << boxes[myk / 6][1] << ' ' << boxes[myk / 6][2] << ' ' << boxes[myk / 6][3] << std::endl;
    // }

    std::cout << std::endl << "total inference time: " << total << std::endl;
    std::cout << "Average running time of one iteration: " << total / static_cast<double>(1) << " ms" << std::endl;
    std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(1) * batchSize / total << " FPS" << std::endl;
    std::cout << std::endl;

    std::cout << "Execution successful" << std::endl;
    return 0;
}

void genprior(float priors[][4]){
    int flag = 0;
    for(int prii = 0; prii < 6; prii ++){
        float scale = (float)pri_image_size / shrinkage[prii];
        for (int prij = 0; prij < feature_map_size[prii]; prij ++){
            for (int prik = 0; prik < feature_map_size[prii]; prik ++){
                float x_center = (prik + 0.5) / scale;
                float y_center = (prij + 0.5) / scale;
                float size = box_sizes_min[prii];
                float h = size/pri_image_size;
                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h; 
                priors[flag][3] = h; 
                flag ++;

                size =sqrt(box_sizes_min[prii]*box_sizes_max[prii]);
                h = size/pri_image_size;
                 
                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h; 
                priors[flag][3] = h; 
                flag ++;

                size = box_sizes_min[prii];
                h = size/pri_image_size;
                float ratio = sqrt(2);
                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h * ratio; 
                priors[flag][3] = h / ratio; 
                flag ++;

                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h / ratio; 
                priors[flag][3] = h * ratio;
                flag ++;

                ratio = sqrt(3);
                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h * ratio; 
                priors[flag][3] = h / ratio; 
                flag ++;

                priors[flag][0] = x_center; 
                priors[flag][1] = y_center; 
                priors[flag][2] = h / ratio; 
                priors[flag][3] = h * ratio;
                flag ++;
            }
        }
    }

    for(int prii = 0; prii < length; prii ++){
        for(int prij = 0; prij < 4; prij ++){
            if(priors[prii][prij] < 0)
                priors[prii][prij] = 0;
            if(priors[prii][prij] > 1)
                priors[prii][prij] = 1;
        }
    }
}
void convert_locations_to_boxes(float boxes[][4],const float priors[][4],const float* detection){
    float buf_boxes[length][4];
    for (int i = 0;i < length; i++){
        buf_boxes[i][0] = detection[i * 6 + 2] * center_variance * priors[i][2] + priors[i][0];
        buf_boxes[i][1] = detection[i * 6 + 3] * center_variance * priors[i][3] + priors[i][1];
        buf_boxes[i][2] = exp(detection[i * 6 + 4] * size_variance) * priors[i][2];
        buf_boxes[i][3] = exp(detection[i * 6 + 5] * size_variance) * priors[i][3];
    }
    for (int i = 0;i < length; i++){
        boxes[i][0] = buf_boxes[i][0] - buf_boxes[i][2] / 2;
        boxes[i][1] = buf_boxes[i][1] - buf_boxes[i][3] / 2;
        boxes[i][2] = buf_boxes[i][0] + buf_boxes[i][2] / 2;
        boxes[i][3] = buf_boxes[i][1] + buf_boxes[i][3] / 2;
    }
}
bool cmp(<std::vector <float>> a,<std::vector <float>> b){
    return a[4] > b[4]
}
//boxes : left_top , right_bottom
float area_of(float a,float b,float c,float d){
    c = (c - a) > 0 ? c - a , 0;
    d = (d - b) > 0 ? d - b , 0;
    return c * d
}
float iou_of(std::vector<float> a, std::vector<float> b){
    float position[4];
    position[0] = a[0] > b[0] ? a , b;
    position[1] = a[1] > b[1] ? a , b;
    position[2] = a[2] < b[2] ? a , b;
    position[3] = a[3] < b[3] ? a , b;

    overlap_area = area_of(position[0],position[1],position[2],position[3]);
    area0 = area_of(a[0],a[1],a[2],a[3]);
    area1 = area_of(b[0],b[1],b[2],b[3]);
    return overlap_area / (area0 + area1 - overlap_area + 1e-5);
}
std::vector<std::vector<float>> hard_nms(std::vector<std::vector<float>> boxes_probs){
    sort(boxes_probs, boxes_probs + boxes_probs.size(), cmp);
    std::vector<std::vector<float>> result;
    while(boxes_probs.size() > 0){
        result.push_back(boxes_probs[0]);
        boxes_probs.pop_front();
        for(i = boxes_probs.size() - 1; i >= 0; i--){
            if(iou_of(boxes_probs[i],result.back()) > nms_iou_threshold)
                boxes_probs.erase(boxes_probs.begin() + i);
        }
    }
    return result;
}