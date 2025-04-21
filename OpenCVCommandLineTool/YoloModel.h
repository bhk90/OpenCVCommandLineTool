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
#include <vector>
#include "MyShape.h"
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS

struct YoloInferenceResult {
    std::vector<MyShape> shapes;
    cv::Mat binary_mask;

    // 使用右值引用和 std::move 的构造函数
    YoloInferenceResult(std::vector<MyShape>&& s, cv::Mat&& b)
        : shapes(std::move(s)), binary_mask(std::move(b)) {}
};

class YoloModel {
public:
    /// ----------------------- 模型初始化与推理 -----------------------
    /// 说明：模型加载、图像推理，返回格式化的 MyShape 结果。

    // 构造函数：加载模型
    YoloModel(const std::string& model_path);

    // 对输入图像进行推理，输出识别到的标注
    void infer(cv::Mat& image);

    const YoloInferenceResult* getInferenceResult() const;

private:
    torch::jit::script::Module model;
    float conf_threshold;
    float nms_threshold;

    std::unique_ptr<YoloInferenceResult> inference_result = nullptr;

    /// ----------------------- 图像预处理与结果可视化 -----------------------
    /// 说明：用于模型推理前的图像预处理（如resize、letterbox）；
    ///      以及将推理结果绘制在图像上的辅助方法。

    // 对图像进行 Letterbox 操作，保持纵横比
    static std::vector<float> Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);

    // 将模型输出的相对框转换为图像中的实际矩形框
    static cv::Rect toBox(const cv::Mat& input, const cv::Rect& range);

    // 将推理结果绘制到图像上（调试或可视化用）
    static void draw_result(cv::Mat& image, std::vector<SegmentOutput>& results, cv::Mat& mask);
};

#endif // YOLOMODEL_H
