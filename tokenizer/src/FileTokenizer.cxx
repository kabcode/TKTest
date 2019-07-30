#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include  <iterator>


int main(int argc, char* argv[])
{
	std::string filename = argv[1];

	std::ifstream filestream(filename, std::ifstream::in);

	if(filestream.is_open())
	{
		std::string line;
		std::vector<std::string> token;
		while (std::getline(filestream, line))
		{
			std::istringstream iss(line);
			token = std::vector<std::string>(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());
			for (auto &tk:token)
			{
				std::cout << tk << std::endl;
			}
		}		
	}

	return EXIT_SUCCESS;

}
