/// ----------------------- Workspace类 -----------------------
/// 
/// 说明：统一管理与图像相关的 **所有操作**
///      确保与图像的交互都通过Workspace进行。
/// 
///		 一、（重要！）Workspace初始化必须按照以下示例：
/// 
///			// 通过图像路径创建一个 MyImage 的 unique_ptr 对象，放入 workspace 中
///			std::unique_ptr<Workspace> workspace; 
///			workspace = std::make_unique<Workspace>("image.jpg");
/// 
///		
///		 二、Workspace的功能：
/// 
///			1. 通过 `getMyImage()` 方法，
///				Workspace能够访问并操作MyImage相关功能。
/// 
///					用法示例：  
///					MyImage& image = workspace.getMyImage();
///					image.someMethod();
///		 
///		 
///			2. 自身功能还包括 标注（MyShape）的增删改、JSON文件的读写、
///				以及通过运行Yolo模型自动生成Shapes。
///		 
/// 
/// ----------------------- Workspace类 -----------------------

#include "Workspace.h"
#include <fstream>
#include <nlohmann/json.hpp> // 需要安装 JSON 库
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

/// ----------------------- 构造函数 -----------------------
Workspace::Workspace(const std::string& image_path)
	: image(std::make_unique<MyImage>(image_path)),
	image_path(image_path),
	annotation_path((fs::path(image_path).parent_path() /
		(fs::path(image_path).stem().string() + ".json")).string()),
	mask_path((fs::path(image_path).parent_path() /
		(fs::path(image_path).stem().string() + "_mask.png")).string()) {
	loadFromAnnotationFile();
	loadFromMaskFile();
}



/// ----------------------- 获取 MyImage 引用 -----------------------
// 获取 MyImage 的引用
MyImage& Workspace::getMyImage() {
	return *image;  // 返回 MyImage 引用
}

// 获取 MyImage 的常量引用
const MyImage& Workspace::getMyImage() const {
	return *image;  // 返回常量引用
}


/// ----------------------- MyShape（标注）的增删改 -----------------------
// 添加标注
void Workspace::addShape(const std::string& label, const std::vector<Point>& points, int shape_type) {
	MyShape shape(label, shape_type);
	shape.setPoints(points);
	shapes.push_back(shape);
}

// 删除标注
bool Workspace::removeShape(size_t index) {
	if (index >= shapes.size()) {
		return false;
	}
	shapes.erase(shapes.begin() + index);
	return true;
}

// 修改标注
bool Workspace::updateShape(size_t index, const std::string& label, const std::vector<Point>& points) {
	if (index >= shapes.size()) {
		return false;
	}
	shapes[index].setLabel(label);
	shapes[index].setPoints(points);
	return true;
}

// 批量添加标注
void Workspace::importShapes(const std::vector<MyShape>& new_shapes) {
	shapes.insert(shapes.end(), new_shapes.begin(), new_shapes.end());
}



/// ----------------------- JSON文件的读写 -----------------------
// 读取标注文件
bool Workspace::loadFromAnnotationFile() {
	if (!fs::exists(annotation_path)) {
		return false; // 标注文件不存在，返回 false
	}

	hasAnnotationFile = true;

	std::ifstream file(annotation_path);
	if (!file) {
		return false;
	}

	json j;
	file >> j;

	if (!j.contains("shapes")) {
		return false;
	}

	shapes.clear();
	for (const auto& shapeJson : j["shapes"]) {
		std::string label = shapeJson["label"];
		int shape_type = shapeJson["shape_type"];
		std::vector<Point> points;

		for (const auto& pointJson : shapeJson["points"]) {
			points.emplace_back(pointJson["x"], pointJson["y"]);
		}

		addShape(label, points, shape_type);
	}

	return true;
}

// 新建标注文件
void Workspace::createAnnotationFile() {
	if (!fs::exists(annotation_path)) {
		saveToAnnotationFile(); // 自动创建新文件
	}
}

// 保存标注文件
bool Workspace::saveToAnnotationFile() const {
	fs::path dir = fs::path(annotation_path).parent_path();
	if (!fs::exists(dir)) {
		fs::create_directories(dir); // 确保目录存在
	}

	json j;
	j["image_path"] = image_path;
	j["shapes"] = json::array();

	for (const auto& shape : shapes) {
		json shapeJson;
		shapeJson["label"] = shape.getLabel();
		shapeJson["shape_type"] = shape.getShapeType();
		shapeJson["points"] = json::array();

		for (const auto& point : shape.getPoints()) {
			shapeJson["points"].push_back({ {"x", point.x}, {"y", point.y} });
		}

		// 获取 SegmentOutput 并存储到 JSON
		const SegmentOutput& segmentOutput = shape.getSegmentOutput();
		shapeJson["segment_output"] = {
			{"id", segmentOutput._id},
			{"confidence", segmentOutput._confidence},
			{"box", {
				{"x", segmentOutput._box.x},
				{"y", segmentOutput._box.y},
				{"width", segmentOutput._box.width},
				{"height", segmentOutput._box.height}
			}}
		};

		// 将 shapeJson 添加到 shapes 数组中
		j["shapes"].push_back(shapeJson);
	}

	std::ofstream file(annotation_path);
	if (!file) {
		return false;
	}
	file << j.dump(4);
	return true;
}


/// ----------------------- PNG掩码图像的读写 -----------------------
// 保存掩码图像为PNG
void Workspace::saveBinaryMaskAsPng() {
	cv::imwrite(mask_path, binary_mask);
}

// 读取PNG掩码图像，放入binary_mask
bool Workspace::loadFromMaskFile() {
	if (!fs::exists(mask_path)) {
		return false; // PNG文件不存在，返回 false 
	}

	hasMaskFile = true;

	cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE); // 以灰度图方式读取
	if (mask.empty()) {
		return false; // 读取失败
	}

	// 转成二值图：像素值为0或255
	cv::threshold(mask, binary_mask, 127, 255, cv::THRESH_BINARY);

	return true;
}


/// ----------------------- Yolo模型相关 -----------------------
// 运行YoloModelProcessor
void Workspace::runYoloModelProcessor(std::shared_ptr<YoloModelProcessor> processor) {
	setYoloModelProcessor(processor);
	
	yolo_model_processor->infer(image->getImageMat());

	importShapes(yolo_model_processor->getShapes());
	binary_mask = yolo_model_processor->getBinaryMask();
}


/// ----------------------- get/set -----------------------
// 获取图像路径
std::string Workspace::getImagePath() const {
	return image_path;
}

// 获取标注文件路径
std::string Workspace::getAnnotationPath() const {
	return annotation_path;
}

// 获取掩码图像路径
std::string Workspace::getMaskPath() const {
	return mask_path;
}

// 获取图像宽度
int Workspace::getImageWidth() const {
	return image->getWidth();
}

// 获取图像高度
int Workspace::getImageHeight() const {
	return image->getHeight();
}

// 获取所有标注
const std::vector<MyShape>& Workspace::getShapes() const {
	return shapes;
}

// 获取 binary_mask 的方法
const cv::Mat& Workspace::getBinaryMask() const {
	return binary_mask;
}

void Workspace::setYoloModelProcessor(std::shared_ptr<YoloModelProcessor> processor) {
	yolo_model_processor = processor;
}