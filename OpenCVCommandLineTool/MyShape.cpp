/// ----------------------- MyShape类 -----------------------
/// 
/// 说明：用于表示单个标注实例（如一个细胞或细菌）；
///      支持多种形状类型（矩形、多边形、掩码），
///      提供点的管理、标签操作、RLE掩码、历史记录等功能；
///      是标注系统中最核心的基础数据结构之一。
/// 
/// ----------------------- MyShape类 -----------------------


// MyShape.cpp
#include "MyShape.h"

MyShape::MyShape(const std::string& label, int shape_type)
    : label(label), shape_type(shape_type) {}

// Getter implementations
const std::vector<Point>& MyShape::getPoints() const {
    return points;
}

const std::string& MyShape::getLabel() const {
    return label;
}

int MyShape::getShapeType() const {
    return shape_type;
}

const std::vector<int>& MyShape::getRLE() const {
    return rle_counts;
}

bool MyShape::hasRLE() const {
    return !rle_counts.empty();
}

// Setter implementations
void MyShape::setPoints(const std::vector<Point>& new_points) {
    saveHistory();
    points = new_points;
}

void MyShape::setLabel(const std::string& new_label) {
    label = new_label;
}

void MyShape::setShapeType(int type) {
    shape_type = type;
}

void MyShape::setRLE(const std::vector<int>& counts) {
    rle_counts = counts;
}

// Point operations
void MyShape::addPoint(double x, double y) {
    saveHistory();
    points.emplace_back(x, y);
}

// History operations
void MyShape::saveHistory() {
    history.push_back(points);
}

void MyShape::undoLastChange() {
    if (!history.empty()) {
        points = history.back();
        history.pop_back();
    }
}

// RLE operations
std::vector<int> MyShape::run_length_encode(const cv::Mat& binary_mask) {
    CV_Assert(binary_mask.type() == CV_8UC1);

    std::vector<int> rle;
    int height = binary_mask.rows;
    int width = binary_mask.cols;

    uchar prev = 0;
    int count = 0;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            uchar pixel = binary_mask.at<uchar>(y, x);
            pixel = pixel > 0 ? 1 : 0; // 二值化：确保只有0或1

            if (pixel != prev) {
                rle.push_back(count);
                count = 1;
                prev = pixel;
            }
            else {
                ++count;
            }
        }
    }
    rle.push_back(count); // 加入最后一段
    return rle;
}

cv::Mat MyShape::rle_decode_to_mask(const std::vector<int>& rle, int height, int width) {
    cv::Mat mask = cv::Mat::zeros(height * width, 1, CV_8UC1);
    int idx = 0;
    uchar val = 0;

    for (int len : rle) {
        if (idx + len > mask.total()) break; // 防止越界
        if (val == 1) {
            std::memset(mask.data + idx, 255, len); // 前景设为255
        }
        idx += len;
        val = 1 - val;
    }

    // COCO 是列优先（column-major），要 reshape 再转置
    cv::Mat reshaped = mask.reshape(1, width).t(); // t() 是转置
    return reshaped.clone(); // 返回的是 rows=height, cols=width 的 CV_8UC1
}

void MyShape::clearRLE() {
    rle_counts.clear();
}
