﻿/// ----------------------- MyShape类 -----------------------
/// 
/// 说明：用于表示单个标注实例（如一个细胞或细菌）；
///      支持多种形状类型（矩形、多边形、掩码），
///      提供点的管理、标签操作、RLE掩码、历史记录等功能；
///      是标注系统中最核心的基础数据结构之一。
/// 
///
/// 
/// ----------------------- MyShape类 -----------------------

#pragma once
#ifndef SHAPE_H
#define SHAPE_H

#include "point.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/// ----------------------- 模型推理结构体 -----------------------
/// 用于存储模型的推理结构体，包括类别、置信度、矩形框及掩码。
struct SegmentOutput {
    int _id;
    float _confidence;
    cv::Rect2f _box;
    cv::Mat _boxMask;
};

class MyShape {
private:
    std::string label;                    // 标签名 (ex. G-, B+, ...)
    int shape_type;                       // 形状类别 (0: rectangle, 1: polygon, 2: mask)
    std::vector<Point> points;            // 形状所有点
    SegmentOutput segment;                // 用于存储模型的推理结构体，包括类别、置信度、矩形框及掩码。
    
    std::vector<std::vector<Point>> history; // 点的历史记录

public:
    MyShape(const std::string& label, int shape_type);

    // Get 方法
    const std::vector<Point>& getPoints() const;
    const std::string& getLabel() const;
    int getShapeType() const;
    const SegmentOutput& getSegmentOutput() const;

    // Set 方法
    void setPoints(const std::vector<Point>& points);
    void setLabel(const std::string& label);
    void setShapeType(int type);
    void setSegmentOutput(const SegmentOutput& segment_output);

    // 点操作
    void addPoint(double x, double y);

    // 历史记录功能
    void saveHistory();
    void undoLastChange();
};

#endif // SHAPE_H
