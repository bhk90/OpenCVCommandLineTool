#include "YoloModel.h"
#include <array>

YoloModel::YoloModel(const std::string& model_path)
    : conf_threshold(0.25f), nms_threshold(0.7f), device(torch::kCPU) {
    model = torch::jit::load(model_path);
    model.to(device);
    model.eval();
}

// 建立ID到名称的映射
static const std::array<std::string, 7> class_id_to_label = {
    "CEC", "RBC", "SEC", "TEC", "TNEC", "TLC", "TMC"
};

struct InputConfig {
    cv::Size letterbox_size; // e.g. (640, 640) or (640, 480)
    int segment_cols;        // 25600 or 19200
    int segment_copy_size;   // 32 * 160 * 160 or 32 * 120 * 160
    int reshape_cols;        // 160 or 120
};

InputConfig getInputConfig(bool is_square) {
    if (is_square) {
        return {
            cv::Size(640, 640),
            25600,
            32 * 160 * 160,
            160
        };
    }
    else {
        return {
            cv::Size(640, 480),
            19200,
            32 * 120 * 160,
            120
        };
    }
}


void YoloModel::infer(cv::Mat& image) {
    bool is_input_square = (image.rows == image.cols);
    InputConfig cfg = getInputConfig(is_input_square);

    cv::Mat resize_image;
    std::vector<float> pad_info = Letterbox(image, resize_image, cfg.letterbox_size);

    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);

    torch::Tensor image_tensor = torch::from_blob(resize_image.data, { resize_image.rows, resize_image.cols, 3 }, torch::kByte).to(device);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({ 2, 0, 1 }).unsqueeze(0);

    std::vector<torch::jit::IValue> inputs{ image_tensor };
    auto net_outputs = model.forward(inputs).toTuple();

    at::Tensor main_output = net_outputs->elements()[0].toTensor().to(torch::kCPU);
    at::Tensor mask_output = net_outputs->elements()[1].toTensor().to(torch::kCPU);

    cv::Mat detect_buffer(main_output.sizes()[1], main_output.sizes()[2], CV_32F, (float*)main_output.data_ptr());
    detect_buffer = detect_buffer.t();

    cv::Mat segment_buffer(32, cfg.segment_cols, CV_32F);
    std::memcpy(segment_buffer.data, mask_output.data_ptr(), sizeof(float) * cfg.segment_copy_size);

    std::vector<cv::Rect> mask_boxes;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Mat> masks;

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
            const cv::Rect mask_box = toBox(detection_box * mask_scale, cv::Rect(0, 0, 160, cfg.reshape_cols));
            const cv::Rect image_box = toBox(detection_box, cv::Rect(0, 0, image.cols, image.rows));
            mask_boxes.push_back(mask_box);
            boxes.push_back(image_box);
            masks.push_back(result.colRange(main_output.sizes()[1] - 32, main_output.sizes()[1]));
        }
    }

    std::vector<int> nms_indexes;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indexes);

    std::vector<SegmentOutput> segmentOutputs;
    std::vector<MyShape> shapes;

    for (const int index : nms_indexes) {
        SegmentOutput segmentOutput;
        segmentOutput._id = class_ids[index];
        segmentOutput._confidence = confidences[index];
        segmentOutput._box = boxes[index];

        cv::Mat m;
        cv::exp(-masks[index] * segment_buffer, m);
        m = 1.0f / (1.0f + m);
        m = m.reshape(1, cfg.reshape_cols);
        cv::resize(m(mask_boxes[index]) > 0.5f, segmentOutput._boxMask, segmentOutput._box.size());

        std::string label = class_id_to_label.at(class_ids[index]);
        const cv::Rect& b = segmentOutput._box;

        MyShape shape(label, 2);
        shape.addPoint(b.x, b.y);
        shape.addPoint(b.x + b.width, b.y + b.height);
        shape.setSegmentOutput(segmentOutput);
        shapes.push_back(shape);

        segmentOutputs.push_back(segmentOutput);
    }

    cv::Mat binary_mask;
    draw_result(resize_image, segmentOutputs, binary_mask);

    // 初始化 unique_ptr， 传入右值
    inference_result = std::make_unique<YoloInferenceResult>(std::move(shapes), std::move(binary_mask));
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

void YoloModel::draw_result(cv::Mat& image, std::vector<SegmentOutput>& results, cv::Mat& mask) {
    mask = cv::Mat::zeros(image.size(), CV_8UC1); // 初始化为黑色

    for (const SegmentOutput& result : results) {
        cv::Rect box = cv::Rect(result._box) & cv::Rect(0, 0, mask.cols, mask.rows);
        if (box.width > 0 && box.height > 0) {
            cv::Mat resizedMask;
            if (result._boxMask.size() != box.size()) {
                cv::resize(result._boxMask, resizedMask, box.size(), 0, 0, cv::INTER_NEAREST);
            }
            else {
                resizedMask = result._boxMask;
            }

            mask(box).setTo(255, resizedMask); // 条件式贴白
        }
    }
}



// 提供访问 inference_result 的方法
const YoloInferenceResult* YoloModel::getInferenceResult() const {
    return inference_result.get();
}