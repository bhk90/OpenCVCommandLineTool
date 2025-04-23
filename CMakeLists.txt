# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("OpenCVCommandLineTool")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "nlohmann_json_DIR = ${nlohmann_json_DIR}")

set(OpenCV_DIR "D:/wh/env/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# set(Torch_DIR "D:/wh/env/libtorch/2.6.0/cpu/debug/share/cmake/Torch")
set(Torch_DIR "D:/wh/env/libtorch/2.6.0/cuda118/debug/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(nlohmann_json REQUIRED)

# 包含子项目。
add_subdirectory ("OpenCVCommandLineTool")
