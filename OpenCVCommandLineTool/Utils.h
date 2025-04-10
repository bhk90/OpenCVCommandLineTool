#pragma once

// Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

std::vector<std::string> parseArguments(const std::string& line);
std::string base64Encode(const unsigned char* data, size_t length);

#endif // UTILS_H
