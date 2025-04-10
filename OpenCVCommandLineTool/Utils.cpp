// Utils.cpp
#include "Utils.h"
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <iterator>

std::vector<std::string> parseArguments(const std::string& line) {
	std::istringstream iss(line);
	std::vector<std::string> args;
	std::string token;

	while (iss >> token) {  // 直接使用流运算符读取
		args.push_back(token);
	}

	return args;
}

