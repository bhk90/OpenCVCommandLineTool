#include <BinaryProcessor.h>


BinaryProcessor::BinaryProcessor(cv::Mat& img)
	: image_mat(img), options(BinaryOptions()) {}


void BinaryProcessor::setOptions(const BinaryOptions& options) {
	this->options = options;
}

#include <opencv2/opencv.hpp>

void BinaryProcessor::makeBinary() {
	if (image_mat.empty()) {
		std::cerr << "Error: Image is empty." << std::endl;
		return;
	}

	// Convert image to grayscale if not already
	if (image_mat.channels() > 1) {
		cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2GRAY);
	}

	// Compute histogram
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;
	cv::calcHist(&image_mat, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

	// Initial threshold estimate: mean pixel value
	double sum = 0, sumForeground = 0, sumBackground = 0;
	int countForeground = 0, countBackground = 0;
	double threshold = cv::mean(image_mat)[0];
	double newThreshold = 0;

	// Iterative threshold calculation
	for (int iter = 0; iter < options.iterations; ++iter) {
		sumForeground = sumBackground = 0;
		countForeground = countBackground = 0;

		for (int i = 0; i < 256; ++i) {
			if (i <= threshold) {
				sumBackground += i * hist.at<float>(i);
				countBackground += hist.at<float>(i);
			}
			else {
				sumForeground += i * hist.at<float>(i);
				countForeground += hist.at<float>(i);
			}
		}

		double meanBackground = (countBackground > 0) ? sumBackground / countBackground : 0;
		double meanForeground = (countForeground > 0) ? sumForeground / countForeground : 0;
		newThreshold = (meanBackground + meanForeground) / 2.0;

		if (std::abs(newThreshold - threshold) < 0.01) break;
		threshold = newThreshold;

		std::cout << "Threshold: " << threshold << ", Mean Background: " << meanBackground
			<< ", Mean Foreground: " << meanForeground << std::endl;
	}

	// Apply threshold
	int thresholdType = options.black_background ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
	cv::threshold(image_mat, image_mat, threshold, 255, thresholdType);

	// Apply optional erosion
	/*if (options.count > 1) {
		int kernelSize = options.count * 2 - 1;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
		cv::erode(image_mat, image_mat, kernel);
	}*/
}


void BinaryProcessor::convertToMask() {
	if (image_mat.empty()) {
		std::cerr << "Error: Image is empty." << std::endl;
		return;
	}

	// Convert image to grayscale if not already
	if (image_mat.channels() > 1) {
		cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2GRAY);
	}

	// Determine threshold
	double threshold = cv::mean(image_mat)[0];
	int thresholdType = options.black_background ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV;

	// Apply threshold
	cv::threshold(image_mat, image_mat, threshold, 255, thresholdType);

	// Convert mask to inverted LUT (black = 255, white = 0)
	image_mat = 255 - image_mat;
}


void BinaryProcessor::erode() {
	cv::erode(image_mat, image_mat, cv::Mat(), cv::Point(-1, -1), options.iterations);
}

void BinaryProcessor::dilate() {
	cv::dilate(image_mat, image_mat, cv::Mat(), cv::Point(-1, -1), options.iterations);
}

void BinaryProcessor::open() {
	cv::morphologyEx(image_mat, image_mat, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), options.iterations);
}

void BinaryProcessor::close() {
	cv::morphologyEx(image_mat, image_mat, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), options.iterations);
}

void BinaryProcessor::median() {
	cv::medianBlur(image_mat, image_mat, 5);
}

void BinaryProcessor::outline() {
	cv::Mat edges;
	cv::Canny(image_mat, edges, 100, 200);
	image_mat = edges;
}

void BinaryProcessor::fillHoles() {
	cv::floodFill(image_mat, cv::Point(0, 0), cv::Scalar(255));
}

void BinaryProcessor::skeletonize() {
	cv::Mat skel(image_mat.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	while (true) {
		cv::erode(image_mat, eroded, element);
		cv::dilate(eroded, temp, element);
		cv::subtract(image_mat, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(image_mat);
		if (cv::countNonZero(image_mat) == 0) break;
	}
	image_mat = skel;
}

void BinaryProcessor::distanceMap() {
	cv::distanceTransform(image_mat, image_mat, cv::DIST_L2, 5);
	cv::normalize(image_mat, image_mat, 0, 1.0, cv::NORM_MINMAX);
}

void BinaryProcessor::ultimatePoints() {
	cv::Mat dist;
	cv::distanceTransform(image_mat, dist, cv::DIST_L2, 5);
	cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
	cv::threshold(dist, image_mat, 0.5, 1.0, cv::THRESH_BINARY);
}

void BinaryProcessor::watershed() {
	cv::Mat markers;
	cv::watershed(image_mat, markers);
	image_mat = markers;
}

void BinaryProcessor::voronoi() {
	cv::Mat dist;
	cv::distanceTransform(image_mat, dist, cv::DIST_L2, 5);
	cv::threshold(dist, dist, 0.5, 1.0, cv::THRESH_BINARY);
	cv::normalize(dist, image_mat, 0, 255, cv::NORM_MINMAX, CV_8U);
}