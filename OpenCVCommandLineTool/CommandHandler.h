#pragma once

// CommandHandler.h
#ifndef COMMAND_HANDLER_H
#define COMMAND_HANDLER_H


#include "MyImage.h"
#include "Workspace.h"


class CommandHandler {
private:
	std::unique_ptr<Workspace> workspace;

public:
	CommandHandler() = default;
	void handleCommand(const std::string& command, const std::vector<std::string>& args);
	void commandHelp() const;
	void commandEcho(const std::vector<std::string>& args);
	void commandShow();
	void commandLoad(const std::vector<std::string>& args);
	void commandUnload();

	void commandExport();

	void commandCrop(const std::vector<std::string>& args);
	void commandScale(const std::vector<std::string>& args);
	void commandFlip(const std::vector<std::string>& args);
	void commandRotate(const std::vector<std::string>& args);
	void commandTranslate(const std::vector<std::string>& args);
	void commandType(const std::vector<std::string>& args);
	void commandSetBrightnessContrast(const std::vector<std::string>& args);

	void commandBinary(const std::vector<std::string>& args);
	void commandSetBinaryOptions(const std::vector<std::string>& args);

	void commandFilter(const std::vector<std::string>& args);

	void commandLabel(const std::vector<std::string>& args);
	void commandModelProcessing(const std::vector<std::string>& args);
};

#endif // COMMAND_HANDLER_H
