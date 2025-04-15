#include "YoloModel.h"

YoloModel::YoloModel(const std::string& model_path)
    : conf_threshold(0.1f), nms_threshold(0.1f) {
    model = torch::jit::load(model_path);
    model.to(torch::kCPU);
    model.eval();
}



YoloInferenceResult YoloModel::infer(cv::Mat& image) {
    // 1. 预处理图像
    cv::Mat resize_image;
    std::vector<float> pad_info = Letterbox(image, resize_image, cv::Size(640, 640));

    torch::Tensor image_tensor = preprocessImage(resize_image);

    // 2. 推理
    auto [main_output, mask_output] = runInference(image_tensor);

    // 3. 结果处理
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Mat> masks;

    processDetections(main_output, mask_output, boxes, class_ids, confidences, masks);

    std::vector<int> nms_indexes;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indexes);

    // 4. 创建 shapes 和 segmentOutputs
    std::vector<SegmentOutput> segmentOutputs;
    std::vector<MyShape> shapes = createShapes(nms_indexes, boxes, class_ids, confidences, masks, segmentOutputs);

    // 5. 绘制结果
    cv::Mat final_result = draw_result(resize_image, segmentOutputs);

    // 6. 返回包含 shapes 和绘制结果的结构体
    YoloInferenceResult result;
    result.shapes = shapes;
    result.draw_result = final_result;

    return result;
}

// 预处理图像
torch::Tensor YoloModel::preprocessImage(const cv::Mat& resize_image) {
    torch::Tensor image_tensor = torch::from_blob(resize_image.data, { resize_image.rows, resize_image.cols, 3 }, torch::kByte).to(torch::kCPU);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({ 2, 0, 1 }).unsqueeze(0);
    return image_tensor;
}

// 推理
std::tuple<at::Tensor, at::Tensor> YoloModel::runInference(const torch::Tensor& image_tensor) {
    std::vector<torch::jit::IValue> inputs{ image_tensor };
    auto net_outputs = model.forward(inputs).toTuple();
    at::Tensor main_output = net_outputs->elements()[0].toTensor().to(torch::kCPU);
    at::Tensor mask_output = net_outputs->elements()[1].toTensor().to(torch::kCPU);
    return { main_output, mask_output };
}

// 处理检测结果
void YoloModel::processDetections(const at::Tensor& main_output, const at::Tensor& mask_output,
    std::vector<cv::Rect>& boxes, std::vector<int>& class_ids,
    std::vector<float>& confidences, std::vector<cv::Mat>& masks) {
    cv::Mat detect_buffer(main_output.sizes()[1], main_output.sizes()[2], CV_32F, (float*)main_output.data_ptr());
    detect_buffer = detect_buffer.t();

    cv::Mat segment_buffer(32, 25600, CV_32F);
    std::memcpy(segment_buffer.data, mask_output.data_ptr(), sizeof(float) * 32 * 160 * 160);

    for (int i = 0; i < detect_buffer.rows; ++i) {
        const cv::Mat result = detect_buffer.row(i);
        const cv::Mat classes_scores = result.colRange(4, main_output.sizes()[1] - 32);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            class_ids.push_back(class_id_point.x);
            confidences.push_back(score);

            const float mask_scale = 0.25f;
            const cv::Mat detection_box = result.colRange(0, 4);
            const cv::Rect mask_box = toBox(detection_box * mask_scale, cv::Rect(0, 0, 160, 160));
            const cv::Rect image_box = toBox(detection_box, cv::Rect(0, 0, image.cols, image.rows));
            boxes.push_back(image_box);
            masks.push_back(result.colRange(main_output.sizes()[1] - 32, main_output.sizes()[1]));
        }
    }
}

// 创建 shapes，并增加对 segmentOutputs 的处理
std::vector<MyShape> YoloModel::createShapes(const std::vector<int>& nms_indexes, const std::vector<cv::Rect>& boxes,
    const std::vector<int>& class_ids, const std::vector<float>& confidences,
    const std::vector<cv::Mat>& masks, std::vector<SegmentOutput>& segmentOutputs) {
    std::vector<MyShape> shapes;

    for (const int index : nms_indexes) {
        SegmentOutput segmentOutput;
        segmentOutput._id = class_ids[index];
        segmentOutput._confidence = confidences[index];
        segmentOutput._box = boxes[index];

        cv::Mat m;
        cv::exp(-masks[index] * segment_buffer, m);
        m = 1.0f / (1.0f + m);
        m = m.reshape(1, 160);
        cv::resize(m(mask_boxes[index]) > 0.5f, segmentOutput._boxMask, segmentOutput._box.size());

        segmentOutputs.push_back(segmentOutput);

        std::string label = std::to_string(class_ids[index]);
        MyShape shape(label, 2);

        const cv::Rect& b = segmentOutput._box;
        shape.addPoint(b.x, b.y);
        shape.addPoint(b.x + b.width, b.y + b.height);

        cv::Mat binary_mask;
        segmentOutput._boxMask.convertTo(binary_mask, CV_8U);
        std::vector<int> rle = shape.run_length_encode(binary_mask);

        shape.setBoundingRect(b);
        shape.setMask(binary_mask);
        shape.setRLE(rle);

        shapes.push_back(shape);
    }

    return shapes;
}

std::vector<float> YoloModel::Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    float in_h = static_cast<float>(src.rows);
    float in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;
    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return { static_cast<float>(left), static_cast<float>(top), scale };
}

cv::Rect YoloModel::toBox(const cv::Mat& input, const cv::Rect& range) {
    float cx = input.at<float>(0);
    float cy = input.at<float>(1);
    float ow = input.at<float>(2);
    float oh = input.at<float>(3);
    cv::Rect box;
    box.x = cvRound(cx - 0.5f * ow);
    box.y = cvRound(cy - 0.5f * oh);
    box.width = cvRound(ow);
    box.height = cvRound(oh);
    return box & range;
}

cv::Mat YoloModel::draw_result(const cv::Mat& image, const std::vector<SegmentOutput>& results) {
    cv::Mat result_image = image.clone();
    cv::Mat mask = image.clone();

    for (const SegmentOutput& result : results) {
        // 绘制边界框
        cv::rectangle(result_image, result._box, cv::Scalar(0, 255, 0), 2, 8);

        // 在mask上应用每个mask
        mask(result._box).setTo(cv::Scalar(0, 0, 255), result._boxMask);
    }

    // 加权合成原图和mask
    cv::Mat final_result;
    cv::addWeighted(result_image, 0.5, mask, 0.8, 1, final_result);
    return final_result;
}

