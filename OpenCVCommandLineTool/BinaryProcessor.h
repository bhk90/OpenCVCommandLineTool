#pragma once

// BinaryProcessor.h
#ifndef BINARY_PROCESSOR_H
#define BINARY_PROCESSOR_H

#include <opencv2/opencv.hpp>

enum EdmOutput {
	kOverwrite,
	k8Bit,
	k16Bit,
	k32Bit
};

struct BinaryOptions {
	int iterations = 1; // 1-100
	int count = 1; // 1-8
	bool black_background = true;
	bool pad_edges_when_eroding = false;
	EdmOutput edm_output = kOverwrite;
};

class BinaryProcessor {
private:
	cv::Mat& image_mat;
	BinaryOptions options;
public:
	BinaryProcessor(cv::Mat& img);
	void setOptions(const BinaryOptions& options);

	// binary 的一系列功能
	void makeBinary(); // 二值化
	void convertToMask(); // 转换为 mask

	void erode(); // 膨胀
	void dilate(); // 腐蚀
	void open(); // 开运算，先腐蚀再膨胀
	void close(); // 闭运算，先膨胀再腐蚀
	void median();

	void outline(); // 提取边缘
	void fillHoles(); // 填洞
	void skeletonize(); // 骨架化

	void distanceMap(); // 欧氏距离映射EDM
	void ultimatePoints(); // 极限腐蚀点
	void watershed(); // 分水岭算法
	void voronoi(); // 维诺图
};

#endif // BINARY_PROCESSOR_H