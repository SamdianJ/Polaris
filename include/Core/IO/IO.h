#ifndef PHYSICSENGINE_IO_H
#define PHYSICSENGINE_IO_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <algorithm>
#include "Polaris.h"
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <fileapi.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

namespace Polaris {

    class FileIO {
    public:
        // // creat directory
        static bool CreateDirectory(const std::string& path) {
#ifdef _WIN32
            return _mkdir(path.c_str()) == 0 || GetLastError() == ERROR_ALREADY_EXISTS;
#else
            return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
        }

        // creat directories
        static bool CreateDirectories(const std::string& path) {
            std::error_code ec;
            return std::filesystem::create_directories(path, ec) || std::filesystem::exists(path);
        }

        // check Directory
        static bool DirectoryExists(const std::string& path) {
            struct stat info;
            return (stat(path.c_str(), &info) == 0) && (info.st_mode & S_IFDIR);
        }

        // list Directory
        static std::vector<std::string> ListDirectory(const std::string& path) {
            std::vector<std::string> entries;
#ifdef _WIN32
            WIN32_FIND_DATA findData;
            HANDLE hFind = FindFirstFile((path + "/*").c_str(), &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    if (strcmp(findData.cFileName, ".") != 0 &&
                        strcmp(findData.cFileName, "..") != 0) {
                        entries.push_back(findData.cFileName);
                    }
                } while (FindNextFile(hFind, &findData));
                FindClose(hFind);
            }
#else
            DIR* dir = opendir(path.c_str());
            if (dir) {
                struct dirent* entry;
                while ((entry = readdir(dir)) != nullptr) {
                    if (strcmp(entry->d_name, ".") != 0 &&
                        strcmp(entry->d_name, "..") != 0) {
                        entries.push_back(entry->d_name);
                    }
                }
                closedir(dir);
            }
#endif
            return entries;
        }

        // ====================== file operations ======================
        static bool FileExists(const std::string& path) {
            struct stat info;
            return (stat(path.c_str(), &info) == 0) && (info.st_mode & S_IFREG);
        }

        static std::string ReadTextFile(const std::string& path) {
            std::ifstream file(path);
            if (!file.is_open())
            {
                PLS_WARN("[Read] failed to open file: {}", path);
                return "";
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        static bool WriteTextFile(const std::string& path,
            const std::string& content,
            bool append = false) {
            std::ofstream file;
            file.open(path, append ? std::ios::app : std::ios::trunc);
            if (!file.is_open())
            {
                PLS_WARN("[Write] failed to open file: {}", path);
                return false;
            }
            file << content;
            return true;
        }

        static std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) return {};

            // 通过文件结尾位置获取文件大小
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            // 读取二进制数据
            std::vector<uint8_t> buffer(size);
            if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
                return buffer;
            }
            return {};
        }

        static bool WriteBinaryFile(const std::string& path,
            const std::vector<uint8_t>& data,
            bool append = false) {
            std::ofstream file(path,
                std::ios::binary | (append ? std::ios::app : std::ios::trunc));
            if (!file.is_open()) return false;

            file.write(reinterpret_cast<const char*>(data.data()), data.size());
            return file.good();
        }

        static bool DeleteFile(const std::string& path) {
            return remove(path.c_str()) == 0;
        }

        static size_t GetFileSize(const std::string& path) {
            struct stat info;
            if (stat(path.c_str(), &info) != 0) return 0;
            return info.st_size;
        }

        static std::string NormalizePath(std::string path) {
#ifdef _WIN32
            std::replace(path.begin(), path.end(), '\\', '/');
#endif
            return path;
        }

        static std::string JoinPath(const std::string& a, const std::string& b) {
            return NormalizePath(a + "/" + b);
        }
    };

}

#endif