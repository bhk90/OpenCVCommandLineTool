#include "YoloModelProcessor.h"

YoloModelProcessor::YoloModelProcessor(const std::string& model_path) {
    yolo_model = std::make_unique<YoloModel>(model_path);
}

void YoloModelProcessor::detectShapes(cv::Mat& image) {
    inference_result = yolo_model->infer(image);
}

// 返回对 shapes 的引用
std::vector<MyShape>& YoloModelProcessor::getShapes() {
    return inference_result.shapes;
}

// 返回绘制结果图像（cv::Mat）
cv::Mat& YoloModelProcessor::getDrawResult() {
    return inference_result.draw_result;
}