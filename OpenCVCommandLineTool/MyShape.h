/// ----------------------- MyShape类 -----------------------
/// 
/// 说明：用于表示单个标注实例（如一个细胞或细菌）；
///      支持多种形状类型（矩形、多边形、掩码），
///      提供点的管理、标签操作、RLE掩码、历史记录等功能；
///      是标注系统中最核心的基础数据结构之一。
/// 
/// ----------------------- MyShape类 -----------------------

#pragma once
#ifndef SHAPE_H
#define SHAPE_H

#include "point.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class MyShape {
private:
    std::string label;                    // 标签名 (ex. G-, B+, ...)
    int shape_type;                       // 形状类别 (0: rectangle, 1: polygon, 2: mask)
    std::vector<Point> points;            // 形状所有点
    std::vector<int> rle_counts;          // 保存 RLE 格式的掩码
    std::vector<std::vector<Point>> history; // 点的历史记录

public:
    MyShape(const std::string& label, int shape_type);

    // Get 方法
    const std::vector<Point>& getPoints() const;
    const std::string& getLabel() const;
    int getShapeType() const;
    const std::vector<int>& getRLE() const;
    bool hasRLE() const;

    // Set 方法
    void setPoints(const std::vector<Point>& points);
    void setLabel(const std::string& label);
    void setShapeType(int type);
    void setRLE(const std::vector<int>& counts);

    // 点操作
    void addPoint(double x, double y);

    // 历史记录功能
    void saveHistory();
    void undoLastChange();

    // RLE 掩码
    std::vector<int> run_length_encode(const cv::Mat& binary_mask);
    cv::Mat rle_decode_to_mask(const std::vector<int>& rle, int height, int width);
    void clearRLE();
};

#endif // SHAPE_H
