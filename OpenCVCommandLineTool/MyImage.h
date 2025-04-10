/// ----------------------- MyImage类 -----------------------
/// 
/// 说明：图像的载体类，封装了 OpenCV 的 Mat 数据结构；
///      提供从路径读取图像、元信息存储、以及图像的裁剪、缩放、变换、增强、分析等功能；
///      并通过 BinaryProcessor、FilterProcessor 对图像进行二值处理与滤波处理；
///      作为 Workspace 中用于标注的基础图像单元。
/// 
///      （重要！）请注意，`MyImage` 对象的创建和管理由 `Workspace` 类统一负责，
///      用户不应直接创建 `MyImage` 实例，而是通过 `Workspace` 来访问和操作图像数据。
///      Workspace 在初始化时会接管 `MyImage` 的所有权，并提供对图像的操作接口。
/// 
/// ----------------------- MyImage类 -----------------------


#pragma once
#ifndef MY_IMAGE_H
#define MY_IMAGE_H

#include <opencv2/opencv.hpp>
#include "BinaryProcessor.h"
#include "FilterProcessor.h"

/// ----------------------- 枚举与元数据结构 -----------------------

/* 枚举色彩位深，用于 MyImage::convertColorDepth() 函数 */
enum ColorDepth {
	k8BitGrayscale,
	k16BitGrayscale,
	k32BitGrayscale,
	k8BitColor,
	kRGBColor
};

/* 图像元数据 */
struct ImageMetadata {
	int dataset_size;              // 数据集大小
	int position_id;               // 位置编号
	std::string imagingDevice;     // 成像设备信息
	cv::Size resolution;           // 图像分辨率
	// 可拓展字段
};

/// ----------------------- 图像类定义 -----------------------

class MyImage {
private:
	std::string image_path;            // 图像路径
	ImageMetadata image_metadata;      // 图像元数据
	cv::Mat image_mat;                 // 图像 Mat 数据

	int image_width;
	int image_height;

public:
	BinaryProcessor binary;            // 二值图处理器
	FilterProcessor filter;            // 滤波图处理器

	/// ----------------------- 构造与基本展示 -----------------------

	// 从路径构造图像
	MyImage(const std::string& image_path);

	// 显示图像（仅用于测试）
	void show() const;

	/// ----------------------- 基本信息获取 -----------------------

	cv::Mat& getImageMat();
	std::string getImagePath() const;
	int getWidth() const;
	int getHeight() const;

	/// ----------------------- 图像导出 -----------------------

	// 导出图像为文件
	void exportImage(std::string outputPath = "temp_image.png");

	/// ----------------------- 图像裁剪 -----------------------

	// 根据给定矩形裁剪图像
	void crop(int x, int y, int width, int height);

	/// ----------------------- 图像缩放 -----------------------

	void scale(float factor);                   // 按比例缩放
	void scaleByWidth(int width);              // 按目标宽度缩放
	void scaleByHeight(int height);            // 按目标高度缩放

	/// ----------------------- 图像几何变换 -----------------------

	void flipHorizontally();                   // 水平翻转
	void flipVertically();                     // 垂直翻转
	void rotateNinetyClockwise();              // 顺时针旋转90度
	void rotateNinetyCounterClockwise();       // 逆时针旋转90度
	void rotate(double angle);                 // 任意角度旋转
	void translate(float x_offset, float y_offset); // 平移图像

	/// ----------------------- 图像类型变换 -----------------------

	void convertColorDepth(ColorDepth color_depth); // 改变颜色深度/模式

	/// ----------------------- 图像调整 -----------------------

	void setBrightnessContrast(int minimum, int maximum, double contrast, double brightness); // 亮度对比度调整
	void threshold(int minimum, int maximum);                                                 // 阈值分割

	/// ----------------------- 图像处理 -----------------------

	void smooth();   // 柔化（平滑）
	void sharpen();  // 锐化

	/// ----------------------- 图像分析 -----------------------

	std::vector<float> histogram();                          // 灰度直方图
	std::vector<float> plotProfile(const cv::Mat& mask);     // 灰度剖面线分析
};

#endif // MY_IMAGE_H
