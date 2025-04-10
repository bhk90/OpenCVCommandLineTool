/// ----------------------- MyShape类 -----------------------
/// 
/// 说明：用于表示单个标注实例（如一个细胞或细菌）；
///      支持多种形状类型（矩形、多边形、掩码），
///      提供点的管理、标签操作、RLE压缩、历史记录等功能；
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
	std::string label;						// 标签名 (ex. G-, B+, ...)
	int shape_type;							// 形状类别 (0: rectangle, 1: polygon, 2: mask)
	std::vector<Point> points;				// 形状所有点

	// 实例分割结果数据（掩码）
	std::vector<int> rle_counts;			// 保存 RLE 格式的掩码

	std::vector<std::vector<Point>> history; // 点的历史记录

public:
	/// ----------------------- 构造与基础设置 -----------------------

	// 构造函数：初始化标签和形状类型
	MyShape(const std::string& label, int shape_type);

	/// ----------------------- 点操作 -----------------------

	// 添加一个点到当前形状
	void addPoint(double x, double y);

	// 获取/设置所有点
	const std::vector<Point>& getPoints() const;
	void setPoints(const std::vector<Point>& points);

	/// ----------------------- 标签与类型操作 -----------------------

	// 获取/设置标签名
	const std::string& getLabel() const;
	void setLabel(const std::string& label);

	// 获取/设置形状类型
	int getShapeType() const;
	void setShapeType(int type);

	/// ----------------------- 历史记录功能 -----------------------

	// 保存当前点列表到历史记录
	void saveHistory();

	// 撤销上一次的点修改
	void undoLastChange();

	/// ----------------------- RLE 操作 -----------------------

	// 根据二值掩码生成 RLE 编码
	std::vector<int> run_length_encode(const cv::Mat& binary_mask);

	// 设置 RLE 数据
	void setRLE(const std::vector<int>& counts);

	// 获取 RLE 数据
	const std::vector<int>& getRLE() const;

	// 清除 RLE 数据
	void clearRLE();

	// 判断是否包含有效 RLE
	bool hasRLE() const;
};

#endif // SHAPE_H
