#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "Polaris.h"

namespace Polaris
{
	enum KernelLaunchStatus
	{
		kSucess,
		kConfigurationError,
		kExecutionError
	};

	struct KernelLaunchConfig
	{
		dim3 grid;
		dim3 block;
		size_t dynamic_smem;

		HOST_DEVICE_FUNC
		KernelLaunchConfig(
			dim3 grid_ = dim3(1, 1, 1),
			dim3 block_ = dim3(1, 1, 1),
			size_t dynamic_smem_ = 0
		)
			:
			grid(grid_),
			block(block_),
			dynamic_smem(dynamic_smem_)
		{}
	};

	
}
