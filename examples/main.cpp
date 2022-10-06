#include <iostream>
#include <vector>
#include <cassert>

#include "image_mosaic.hpp"
#include "parser.hpp"

int main(int argc, char *argv[])
{
	assert(argc == 2);
	std::vector<std::string> names = parser(argv[1]);
	ImageMosaic image_mosaic(names);
	image_mosaic.mosaic();
}
