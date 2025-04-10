#include "FilterProcessor.h"

FilterProcessor::FilterProcessor(cv::Mat& img) : image_mat(img) {}


void FilterProcessor::convolve(const std::string& kernel_str) {
	// Parse the kernel string into a cv::Mat
	std::vector<float> kernel_values;
	std::stringstream ss(kernel_str);
	std::string line;
	int rows = 0;
	while (std::getline(ss, line)) {
		std::stringstream line_ss(line);
		float value;
		while (line_ss >> value) {
			kernel_values.push_back(value);
		}
		rows++;
	}

	// Assuming square kernel, calculate the size
	int cols = kernel_values.size() / rows;
	if (cols * rows != kernel_values.size()) {
		std::cerr << "Invalid kernel format!" << std::endl;
		return;
	}

	// Create cv::Mat from the parsed values
	cv::Mat kernel(rows, cols, CV_32F, kernel_values.data());

	// Apply the kernel to the image
	cv::filter2D(image_mat, image_mat, -1, kernel);
}

void FilterProcessor::gaussianBlur(float sigma) {
	cv::GaussianBlur(image_mat, image_mat, cv::Size(0, 0), sigma);
}

void FilterProcessor::median(float radius) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::medianBlur(image_mat, image_mat, kernel_size);
}

void FilterProcessor::mean(float radius) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::blur(image_mat, image_mat, cv::Size(kernel_size, kernel_size));
}

void FilterProcessor::minimum(float radius) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
	cv::morphologyEx(image_mat, image_mat, cv::MORPH_ERODE, element);
}

void FilterProcessor::maximum(float radius) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
	cv::morphologyEx(image_mat, image_mat, cv::MORPH_DILATE, element);
}

void FilterProcessor::unsharpMask(float radius, float mask_weight) {
	cv::Mat blurred;
	cv::GaussianBlur(image_mat, blurred, cv::Size(0, 0), radius);

	cv::addWeighted(image_mat, 1.0 + mask_weight, blurred, -mask_weight, 0, image_mat);
}

void FilterProcessor::variance(float radius) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::Mat mean, squared_mean, variance;

	cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (kernel_size * kernel_size);
	cv::filter2D(image_mat, mean, -1, kernel);
	cv::filter2D(image_mat.mul(image_mat), squared_mean, -1, kernel);

	variance = squared_mean - mean.mul(mean);

	image_mat = variance;
}

void FilterProcessor::topHat(float radius, bool light_background, bool dont_subtract) {
	int kernel_size = static_cast<int>(2 * radius + 1);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));

	cv::Mat result;

	if (light_background) {
		cv::morphologyEx(image_mat, result, cv::MORPH_CLOSE, element);
		if (!dont_subtract) {
			result = result - image_mat;
		}
	}
	else {
		cv::morphologyEx(image_mat, result, cv::MORPH_OPEN, element);
		if (!dont_subtract) {
			result = result - image_mat;
		}
	}
	image_mat = result;
}

void FilterProcessor::showCircularMasks() {
	cv::Mat mask = cv::Mat::zeros(image_mat.size(), CV_8UC1);
	cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), 50, cv::Scalar(255), -1);
	cv::imshow("Circular Mask", mask);
	cv::waitKey(0);
}