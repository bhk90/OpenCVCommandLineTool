/// ----------------------- YoloModelProcessor类 -----------------------
/// 
/// 说明：作为YoloModel的进一步封装，负责将模型推理结果转换为可用的Shape结构；
///      主要提供一种更简单的接口，用于对图像进行目标检测；
///      与 Workspace 协同工作，用于自动生成标注（Shape）。
///         ↑（参见 Workspace 中的 initYoloModelProcessor 和 runYoloOnImage）
/// 
/// ----------------------- YoloModelProcessor类 -----------------------

#ifndef YOLOMODEL_PROCESSOR_H
#define YOLOMODEL_PROCESSOR_H

#include <vector>
#include <string>
#include "YoloModel.h"
#include "MyShape.h"
#include "MyImage.h"

class YoloModelProcessor {
private:
    std::unique_ptr<YoloModel> yolo_model;

public:
    /// ----------------------- 构造与推理 -----------------------
    /// 说明：构造时加载模型，使用 detectShapes 对图像进行目标检测。

    // 构造函数：加载指定路径的模型
    YoloModelProcessor(const std::string& model_path);

    // 对图像执行推理，返回转换为 MyShape 的结果列表
    std::vector<MyShape> detectShapes(cv::Mat& image);
};

#endif // YOLOMODEL_PROCESSOR_H
