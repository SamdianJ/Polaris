#include "Test/Core/MatrixTest.h"
#include <iostream>

using namespace Polaris;

void Polaris::MatrixTest()
{
	Matrix<Scalar, Platform::CPU> matrix(4, 4);

	Matrix<Scalar, Platform::CPU> matrix2;
	matrix2.Transfer(matrix);
	std::cout << matrix << std::endl;
	std::cout << matrix2 << std::endl;
}