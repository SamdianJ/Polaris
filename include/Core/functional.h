#pragma once

#include "Polaris.h"

namespace Polaris
{
	template <typename T>
	struct abs_op 
	{
		HOST_DEVICE_FUNC T operator()(T lhs)
		{
			return (lhs > 0 ? lhs : -lhs);
		}
	};
}