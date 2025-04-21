#include "YoloModelProcessor.h"

YoloModelProcessor::YoloModelProcessor(const std::string& model_path) {
    yolo_model = std::make_unique<YoloModel>(model_path);
}

void YoloModelProcessor::infer(cv::Mat& image) {
    if (yolo_model) {
        yolo_model->infer(image);
    }

}

// 获取检测到的形状
std::vector<MyShape> YoloModelProcessor::getShapes() const {
    if (yolo_model) {
        const YoloInferenceResult* result = yolo_model->getInferenceResult();
        if (result) {
            return result->shapes;
        }
    }
    return {};
}

// 获取二进制掩码
const cv::Mat& YoloModelProcessor::getBinaryMask() const {
    if (yolo_model) {
        const YoloInferenceResult* result = yolo_model->getInferenceResult();
        if (result) {
            return result->binary_mask;
        }
    }
    return cv::Mat();
}