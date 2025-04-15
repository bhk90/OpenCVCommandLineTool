/// ----------------------- YoloModel类 -----------------------
/// 
/// 说明：封装YOLO模型的加载与推理操作；
///      使用LibTorch加载YOLO模型，并利用OpenCV进行图像预处理；
///      负责将输入图像送入模型，输出一系列 MyShape 对象（标注结果）；
///      不负责标注的管理或显示，仅处理模型相关逻辑。
/// 
/// ----------------------- YoloModel类 -----------------------

#ifndef YOLOMODEL_H
#define YOLOMODEL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include "MyShape.h"

/// ----------------------- 模型推理结构体 -----------------------
/// 用于存储模型的中间输出结果，包括类别、置信度、矩形框及掩码。
struct SegmentOutput {
    int _id;
    float _confidence;
    cv::Rect2f _box;
    cv::Mat _boxMask;
};

struct YoloInferenceResult {
    std::vector<MyShape> shapes;
    cv::Mat draw_result;
};

class YoloModel {
public:
    /// ----------------------- 模型初始化与推理 -----------------------
    /// 说明：模型加载、图像推理，返回格式化的 MyShape 结果。

    // 构造函数：加载模型
    YoloModel(const std::string& model_path);

    // 对输入图像进行推理，输出识别到的标注
    YoloInferenceResult infer(cv::Mat& image);


private:
    torch::jit::script::Module model;
    float conf_threshold;
    float nms_threshold;

    /// ----------------------- 图像预处理与结果可视化 -----------------------
    /// 说明：用于模型推理前的图像预处理（如resize、letterbox）；
    ///      以及将推理结果绘制在图像上的辅助方法。
    
    torch::Tensor preprocessImage(const cv::Mat& resize_image);
    std::tuple<at::Tensor, at::Tensor> runInference(const torch::Tensor& image_tensor);
    void processDetections(const at::Tensor& main_output, const at::Tensor& mask_output,
        std::vector<cv::Rect>& boxes, std::vector<int>& class_ids,
        std::vector<float>& confidences, std::vector<cv::Mat>& masks);
    std::vector<MyShape> createShapes(const std::vector<int>& nms_indexes, const std::vector<cv::Rect>& boxes,
        const std::vector<int>& class_ids, const std::vector<float>& confidences,
        const std::vector<cv::Mat>& masks, std::vector<SegmentOutput>& segmentOutputs);

    // 对图像进行 Letterbox 操作，保持纵横比
    static std::vector<float> Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);

    // 将模型输出的相对框转换为图像中的实际矩形框
    static cv::Rect toBox(const cv::Mat& input, const cv::Rect& range);

    // 将推理结果绘制到图像上（调试或可视化用）
    static void draw_result(cv::Mat& image, std::vector<SegmentOutput>& results);
};

#endif // YOLOMODEL_H
