#include "YoloModelProcessor.h"

YoloModelProcessor::YoloModelProcessor(const std::string& model_path) {
    yolo_model = std::make_unique<YoloModel>(model_path);
}

std::vector<MyShape> YoloModelProcessor::detectShapes(cv::Mat& image) {
    return yolo_model->infer(image);
}