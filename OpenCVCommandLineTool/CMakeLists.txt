# CMakeList.txt: OpenCVCommandLineTool 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

file(GLOB SOURCES
	"Main.cpp"
	"Workspace.cpp"
	"MyImage.cpp"
	"MyShape.cpp"
	"BinaryProcessor.cpp"
	"FilterProcessor.cpp"
	"YoloModel.cpp"
	"YoloModelProcessor.cpp"
	"Utils.cpp"
	"CommandHandler.cpp"
	#"ModelProcessor.cpp"
)

# 将源代码添加到此项目的可执行文件。
add_executable (OpenCVCommandLineTool ${SOURCES} "YoloModelProcessor.h")
target_link_libraries(OpenCVCommandLineTool ${OpenCV_LIBS} ${TORCH_LIBRARIES} nlohmann_json)

# 添加头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"  # 当前源文件夹，包含所有 .h 文件
)

# TODO: 如有需要，请添加测试并安装目标。
