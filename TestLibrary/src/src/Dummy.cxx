#include "Dummy.h"
#include <iostream>

Dummy::Dummy(std::string name):
m_name(name)
{
	this->m_ImagePointer = itk::Image<float, 3>::New();
}

Dummy::~Dummy(){}

void Dummy::Print() const
{
	std::cout << m_name << std::endl;
}

std::string Dummy::GetName() const
{
	return this->m_name;
}

void Dummy::SetName(std::string name)
{
	this->m_name = name;
}

