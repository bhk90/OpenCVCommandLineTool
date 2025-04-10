#include "CommandHandler.h"
#include "Utils.h"

int main() {
	CommandHandler command_handler;
	std::string commandLine;

	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	while (true) {
		std::cout << "> ";
		std::getline(std::cin, commandLine);

		if (commandLine.empty()) continue;

		std::vector<std::string> args = parseArguments(commandLine);
		if (args.empty()) continue;

		std::string command = args[0];
		args.erase(args.begin());

		command_handler.handleCommand(command, args);
	}

	return 0;
}