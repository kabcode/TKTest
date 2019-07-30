#include <omp.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[])
{

	const auto arrSize = 52428800; // entries for the initial array
	auto arr = new double[arrSize * 3]; // 3 sequential elements are a group

    #pragma omp parallel for
	for( auto i = 0; i < arrSize*3; ++i)
	{
		arr[i] = 1.0*i/arrSize;
	}

	auto larr = new double[arrSize * 4]; // larger array with space for interleaving elements

    #pragma omp parallel
    #pragma omp for
	for (auto i = 0; i < arrSize; ++i) // after each element group of three a new element should be inserted
	{
		auto k = 0;
		auto j = i*3;
		for(k=0; k < 3; ++k)
		{
			larr[i*4+k] = arr[j+k];
		}
		larr[i*4+k] = 0;
	}

	delete[] arr;
	delete[] larr;

	/* Pointer array test*/
	double array[] = { 0,1,2,3,4,5};

	for (auto &i : array)
	{
		std::cout << i << std::endl;
	}
	


	return EXIT_SUCCESS;
}
