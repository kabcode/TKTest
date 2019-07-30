#ifndef DUMMY_H
#define DUMMY_H

#include <string>
#include "itkImage.h"
#include "testlibrary_export.h"

class TESTLIB_API Dummy
{
	explicit Dummy(std::string name);
	~Dummy();
	
	void Print() const;
	std::string GetName() const;
	void SetName(std::string);
	
	private:
		std::string m_name;
		itk::Image<float, 3>::Pointer m_ImagePointer;
};

#endif

