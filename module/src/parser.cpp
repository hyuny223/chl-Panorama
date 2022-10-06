#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

#include "parser.hpp"

std::vector<std::string> parser(const std::string& p)
{
    std::filesystem::path path(p);
    std::filesystem::directory_iterator itr(path);

    std::vector<std::string> files;

    while(itr != std::filesystem::end(itr))
    {
        const std::filesystem::directory_entry& entry = *itr;

        if(!std::filesystem::is_directory(entry))
        {
            std::cout << entry.path() << std::endl;
            files.emplace_back(entry.path());
        }

        ++itr;
    }

    std::sort(files.begin(), files.end());

    return files;
}
