#pragma once

#ifndef FILTER_PROCESSOR_H
#define FILTER_PROCESSOR_H

#include <opencv2/opencv.hpp>

class FilterProcessor {
private:
	cv::Mat& image_mat;  // 引用 Image 的 Mat，避免数据复制

public:
	FilterProcessor(cv::Mat& img);

	void convolve(const std::string& kernel_str);
	void gaussianBlur(float sigma);
	void median(float radius);
	void mean(float radius);
	void minimum(float radius);
	void maximum(float radius);
	void unsharpMask(float radius, float mask_weight);
	void variance(float radius);
	void topHat(float radius, bool light_background, bool dont_subtract);
	void showCircularMasks();
};

#endif // FILTER_PROCESSOR_H