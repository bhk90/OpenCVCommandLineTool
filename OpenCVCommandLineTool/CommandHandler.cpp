#include "CommandHandler.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <map>

void CommandHandler::handleCommand(const std::string& command, const std::vector<std::string>& args) {
	if (command == "help") {
		commandHelp();
	}
	else if (command == "echo") {
		commandEcho(args);
	}
	else if (command == "load") {
		commandLoad(args);
	}
	else if (command == "unload") {
		commandUnload();
	}
	else if (!workspace) {
		std::cout << "Error: No image loaded. Use 'load <image_path>' first.\n";
	}
	else if (command == "export") {
		commandExport();
	}
	else if (command == "show") {
		commandShow();
	}
	else if (command == "label") {
		commandLabel(args);
	}
	else if (command == "model") {
		commandModelProcessing(args);
	}
	else if (command == "crop") {
		commandCrop(args);
	}
	else if (command == "scale") {
		commandScale(args);
	}
	else if (command == "flip") {
		commandFlip(args);
	}
	else if (command == "rotate") {
		commandRotate(args);
	}
	else if (command == "translate") {
		commandTranslate(args);
	}
	else if (command == "type") {
		commandType(args);
	}
	else if (command == "set_brightness_contrast") {
		commandSetBrightnessContrast(args);
	}
	else if (command == "binary") {
		commandBinary(args);
	}
	else if (command == "filter") {
		commandFilter(args);
	}
	else if (command == "quit") {
		std::cout << "Exiting the program..." << std::endl;
		exit(0);
	}
	else {
		std::cout << "Unknown command: " << command << std::endl;
	}
}

void CommandHandler::commandHelp() const {
	std::cout << "Available commands:\n"
		<< "  help                          - Display this help message\n"
		<< "  echo <arg>                    - Echo the argument back to you\n"
		<< "  load <path/to/image>          - Load an image file\n"
		<< "  unload                        - Unload the current image\n"
		<< "  export                        - Export the image as binary stream\n"
		<< "  show                          - Show image (For testing purpose)"
		<< "  model <path/to/model>         - Use model to generate a JSON annotation file\n"
		<< "  label list					- list all labels\n"
		<< "  label add SEC 0 x0 y0 x1 y1   - add an SEC rectangle label\n"
		<< "  crop <x> <y> <width> <height> - Crop the image\n"
		<< "  scale <factor>                - Scale the image by factor. (Example: scale 0.65)\n"
		<< "  scale width|height <length>   - Scale the image by setting new width/height. (Example: scale width 230)\n"
		<< "  flip h                        - Flip image horizontally\n"
		<< "  flip v                        - Flip image vertically\n"
		<< "  rotate <angle>                - Rotate the image by specified angle\n"
		<< "  translate <x_offset> <y_offset>   - Translate the image\n"
		<< "  type 8bit_gray|16bit_gray|32bit_gray  - Convert color depth\n"
		<< "       |8bit_color|rgb_color\n"
		<< "  set_brightness_contrast       - Set minimum, maximum, brightness and contrast values."
		<< "        [brightness <-127~127>]\n"
		<< "        [contrast <default=1.0>]\n"
		<< "        [minimum <0~255>]\n"
		<< "        [maximum <0~255>]\n"
		<< "  binary                        - Binary\n"
		<< "  filter                        - Filter\n"
		<< "  quit                          - Exit the program\n";
}

void CommandHandler::commandEcho(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'echo' requires an argument.\n";
	}
	else {
		std::cout << args[0] << std::endl;
	}
}

void CommandHandler::commandLoad(const std::vector<std::string>& args) {
	if (args.empty() || args[0].empty()) {
		std::cout << "Error: 'load' requires a valid image path.\n";
		return;
	}

	const std::string& path = args[0];
	if (!std::filesystem::exists(path)) {
		std::cout << "Error: Path '" << path << "' does not exist.\n";
		return;
	}

	if (!std::filesystem::is_regular_file(path)) {
		std::cout << "Error: '" << path << "' is not a valid file.\n";
		return;
	}

	static const std::set<std::string> valid_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
	if (valid_extensions.count(std::filesystem::path(path).extension().string()) == 0) {
		std::cout << "Error: Unsupported file format. Supported formats are: .jpg, .jpeg, .png, .bmp, .tiff\n";
		return;
	}

	try {
		workspace = std::make_unique<Workspace>(std::make_unique<MyImage>(path));
		std::cout << "Image loaded successfully: " << path << std::endl;
	}
	catch (const std::exception& e) {
		std::cout << "Error: Failed to load image. " << e.what() << "\n";
	}
}


void CommandHandler::commandUnload() {
	if (!workspace) {
		std::cout << "No image loaded.\n";
	}
	else {
		workspace.reset();
		std::cout << "Image unloaded successfully.\n";
	}
}

// 仅用于测试
void CommandHandler::commandShow() {
	workspace->getMyImage().show();
}

void CommandHandler::commandExport() {
	workspace->getMyImage().exportImage();
}


// 标注处理
void CommandHandler::commandLabel(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'label' requires at least 1 argument.\n";
		return;
	}
	if (args[0] == "list") {
		std::vector<MyShape> shapes = workspace->getShapes();
		int index = 0;
		for (auto& shape : shapes) {
			
			std::cout << index << ". " << "Label: " + shape.getLabel() + ", Points: [";

			auto& points = shape.getPoints();
			for (auto& point : points) {
				std::cout << "(" << point.x << ", " << point.y << "), ";
			}

			std::cout << "]\n";

			index++;
		}
	}
	if (args[0] == "add") {
		if (args.size() < 3) {
			std::cout << "Error: 'label add' requires at least 3 arguments (label, shape_type, and points).\n";
			return;
		}

		std::string label = args[1];
		int shape_type;
		try {
			shape_type = std::stoi(args[2]);
		}
		catch (const std::exception&) {
			std::cout << "Error: Invalid shape_type, must be an integer.\n";
			return;
		}

		std::vector<Point> points;
		for (size_t i = 3; i + 1 < args.size(); i += 2) {
			try {
				int x = std::stoi(args[i]);
				int y = std::stoi(args[i + 1]);
				points.emplace_back(x, y);
			}
			catch (const std::exception&) {
				std::cout << "Error: Invalid point coordinates, must be integers.\n";
				return;
			}
		}

		if (points.empty()) {
			std::cout << "Error: At least one point is required.\n";
			return;
		}

		workspace->addShape(label, points, shape_type);
		workspace->saveToAnnotationFile();
		std::cout << "Shape added successfully.\n";
	}
	if (args[0] == "remove") {
		if (args.size() < 2) {
			std::cout << "Error: 'label remove' requires 1 argument: index.\n";
			return;
		}
		int index;
		try {
			index = std::stoi(args[1]);
		}
		catch (const std::exception&) {
			std::cout << "Error: Invalid index, must be an integer.\n";
			return;
		}
		workspace->removeShape(index);
		std::cout << "Successfully removed index " << index << ".\n";
	}
}

// 模型预测
void CommandHandler::commandModelProcessing(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'model' requires 1 argument: model_path\n";
		return;
	}
	if (args.size() == 1) {
		workspace->initYoloModelProcessor(args[0]);
		workspace->runYoloOnImage();
		workspace->saveToAnnotationFile();
	}
}


void CommandHandler::commandCrop(const std::vector<std::string>& args) {
	if (args.size() != 4) {
		std::cout << "Error: 'crop' requires 4 arguments (x, y, width, height).\n";
		return;
	}
	workspace->getMyImage().crop(std::stoi(args[0]), std::stoi(args[1]), std::stoi(args[2]), std::stoi(args[3]));
}

void CommandHandler::commandScale(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'scale' requires at least 1 argument.\n";
		return;
	}
	if (args.size() == 1) {
		workspace->getMyImage().scale(std::stod(args[0]));
	}
	else if (args.size() == 2) {
		if (args[0] == "width") {
			workspace->getMyImage().scaleByWidth(std::stod(args[1]));
		}
		else if (args[0] == "height") {
			workspace->getMyImage().scaleByHeight(std::stod(args[1]));
		}
		else {
			std::cout << "Error: Invalid argument for 'scale'. Use 'scale width <int>' or 'scale height <int>'.\n";
		}
	}
}

void CommandHandler::commandFlip(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'flip' requires 'h' or 'v' as an argument.\n";
		return;
	}
	if (args[0] == "h") {
		workspace->getMyImage().flipHorizontally();
	}
	else if (args[0] == "v") {
		workspace->getMyImage().flipVertically();
	}
	else {
		std::cout << "Error: Invalid argument for 'flip'. Use 'flip h' or 'flip v'.\n";
	}
}

void CommandHandler::commandRotate(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'rotate' requires an angle.\n";
		return;
	}
	if (args[0] == "90") {
		workspace->getMyImage().rotateNinetyClockwise();
	}
	else if (args[0] == "-90") {
		workspace->getMyImage().rotateNinetyCounterClockwise();
	}
	else {
		try {
			workspace->getMyImage().rotate(std::stod(args[0]));
		}
		catch (const std::invalid_argument&) {
			std::cout << "Error: Invalid angle format. Use numbers like 'rotate 45'.\n";
		}
	}
}

void CommandHandler::commandTranslate(const std::vector<std::string>& args) {
	if (args.size() != 2) {
		std::cout << "Error: 'translate' requires 2 arguments (x_offset, y_offset).\n";
		return;
	}
	workspace->getMyImage().translate(std::stoi(args[0]), std::stoi(args[1]));
}

void CommandHandler::commandType(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'type' requires an argument (e.g., '8bit_gray', '16bit_gray', '32bit_gray', '8bit_color', 'rgb_color')\n";
		return;
	}

	const std::string& type = args[0];
	ColorDepth depth;

	if (type == "8bit_gray") depth = k8BitGrayscale;
	else if (type == "16bit_gray") depth = k16BitGrayscale;
	else if (type == "32bit_gray") depth = k32BitGrayscale;
	else if (type == "8bit_color") depth = k8BitColor;
	else if (type == "rgb_color") depth = kRGBColor;
	else {
		std::cout << "Error: Invalid type '" << type << "'. Valid types: 8bit_gray, 16bit_gray, 32bit_gray, 8bit_color, rgb_color.\n";
		return;
	}

	workspace->getMyImage().convertColorDepth(depth);
}

void CommandHandler::commandSetBrightnessContrast(const std::vector<std::string>& args) {
	int minimum = 0, maximum = 255;
	double contrast = 0, brightness = 0;

	// 如果 args 为空，输出错误
	if (args.empty()) {
		std::cout << "Error: 'set_brightness_contrast' requires arguments.\n";
		return;
	}

	// 遍历参数并验证配对
	for (size_t i = 0; i < args.size(); ++i) {
		// 如果当前参数是属性名
		if (args[i] == "min" || args[i] == "max" || args[i] == "contrast" || args[i] == "brightness") {
			// 确保属性名后有一个值
			if (i + 1 < args.size()) {
				// 处理对应的属性值
				try {
					if (args[i] == "min") {
						minimum = std::stoi(args[i + 1]);
					}
					else if (args[i] == "max") {
						maximum = std::stoi(args[i + 1]);
					}
					else if (args[i] == "contrast") {
						contrast = std::stod(args[i + 1]);
					}
					else if (args[i] == "brightness") {
						brightness = std::stod(args[i + 1]);
					}
					++i; // 跳过属性值
				}
				catch (const std::invalid_argument& e) {
					std::cout << "Error: Invalid value for property: " << args[i] << std::endl;
					return;
				}
				catch (const std::out_of_range& e) {
					std::cout << "Error: Value out of range for property: " << args[i] << std::endl;
					return;
				}
			}
			else {
				// 如果属性名后没有值，则报错
				std::cout << "Error: Missing value for property: " << args[i] << std::endl;
				return;
			}
		}
		else {
			// 如果遇到无效的参数名，则报错
			std::cout << "Error: Invalid argument: " << args[i] << std::endl;
			return;
		}
	}

	// 调用图像处理函数
	workspace->getMyImage().setBrightnessContrast(minimum, maximum, contrast, brightness);
}

void CommandHandler::commandBinary(const std::vector<std::string>& args) {
	if (args.size() != 1) {
		std::cout << "Error: 'binary' requires 1 argument.\n";
		return;
	}
	if (args[0] == "make") {
		workspace->getMyImage().binary.makeBinary();
	}
	else if (args[0] == "mask") {
		workspace->getMyImage().binary.convertToMask();
	}
	else if (args[0] == "erode") {
		workspace->getMyImage().binary.erode();
	}
	else if (args[0] == "dilate") {
		workspace->getMyImage().binary.dilate();
	}
	else if (args[0] == "open") {
		workspace->getMyImage().binary.open();
	}
	else if (args[0] == "close") {
		workspace->getMyImage().binary.close();
	}
	else if (args[0] == "median") {
		workspace->getMyImage().binary.median();
	}
	else if (args[0] == "outline") {
		workspace->getMyImage().binary.outline();
	}
	else if (args[0] == "fill_holes") {
		workspace->getMyImage().binary.fillHoles();
	}
	else if (args[0] == "skeletonize") {
		workspace->getMyImage().binary.skeletonize();
	}
	else if (args[0] == "distance_map") {
		workspace->getMyImage().binary.distanceMap();
	}
	else if (args[0] == "ultimate_points") {
		workspace->getMyImage().binary.ultimatePoints();
	}
	else if (args[0] == "watershed") {
		workspace->getMyImage().binary.watershed();
	}
	else if (args[0] == "voronoi") {
		workspace->getMyImage().binary.voronoi();
	}
}


void CommandHandler::commandFilter(const std::vector<std::string>& args) {
	if (args.empty()) {
		std::cout << "Error: 'filter' requires at least 1 argument.\n";
		return;
	}
	/*if (args[0] == "convolve") {
		workspace->getMyImage().filter.convolve();
	}*/
	if (args[0] == "gaussian") {
		workspace->getMyImage().filter.gaussianBlur(2);
	}
	if (args[0] == "median") {
		workspace->getMyImage().filter.median(2);
	}
	if (args[0] == "mean") {
		workspace->getMyImage().filter.mean(2);
	}
	if (args[0] == "minimum") {
		workspace->getMyImage().filter.minimum(2);
	}
	if (args[0] == "maximum") {
		workspace->getMyImage().filter.maximum(2);
	}
	if (args[0] == "unsharp") {
		workspace->getMyImage().filter.unsharpMask(1, 0.6);
	}
	if (args[0] == "variance") {
		workspace->getMyImage().filter.variance(2);
	}
	if (args[0] == "tophat") {
		workspace->getMyImage().filter.topHat(2, true, true);
	}
}