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
    std::vector<int> rle;
    int count = 0;
    uchar prev = 0;

    for (int i = 0; i < binary_mask.total(); ++i) {
        uchar pixel = binary_mask.data[i];
        if (pixel != prev) {
            rle.push_back(count);
            count = 1;
            prev = pixel;
        }
        else {
            ++count;
        }
    }
    rle.push_back(count);
    return rle;
}

void MyShape::clearRLE() {
    rle_counts.clear();
}
