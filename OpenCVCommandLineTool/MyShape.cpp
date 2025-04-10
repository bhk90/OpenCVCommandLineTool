// MyShape.cpp
#include "MyShape.h"

MyShape::MyShape(const std::string& label, int shape_type)
	: label(label), shape_type(shape_type) {}

void MyShape::addPoint(double x, double y) {
	saveHistory();
	points.emplace_back(x, y);
}

void MyShape::setPoints(const std::vector<Point>& new_points) {
	saveHistory();
	points = new_points;
}

const std::string& MyShape::getLabel() const {
	return label;
}

void MyShape::setLabel(const std::string& new_label) {
	label = new_label;
}

int MyShape::getShapeType() const {
	return shape_type;
}

void MyShape::setShapeType(int type) {
	shape_type = type;
}

const std::vector<Point>& MyShape::getPoints() const {
	return points;
}

void MyShape::saveHistory() {
	history.push_back(points);
}

void MyShape::undoLastChange() {
	if (!history.empty()) {
		points = history.back();
		history.pop_back();
	}
}

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

void MyShape::setRLE(const std::vector<int>& counts) {
	rle_counts = counts;
}
const std::vector<int>& MyShape::getRLE() const {
	return rle_counts;
}
void MyShape::clearRLE() {
	rle_counts.clear();
}
bool MyShape::hasRLE() const {
	return !rle_counts.empty();
}