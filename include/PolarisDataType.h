#pragma once

#include <cstdint>
#include <limits>
#include <cmath>

namespace Polaris {

	// 基本整数类型
	using Label = std::uint32_t;
	using PlsInt32 = std::int32_t;
	using PlsInt64 = std::int64_t;
	using PlsInt8 = std::int8_t;
	using PlsUInt8 = std::uint8_t;
	using PlsUInt32 = std::uint32_t;
	using PlsUInt64 = std::uint64_t;

	// 浮点数类型
	using PlsFloat = float;
	using PlsDouble = double;

	// 数学常量
	constexpr double PI = 3.14159265358979323846;
	constexpr double TwoPI = 2.0 * PI;
	constexpr double HalfPI = PI / 2.0;
	constexpr double InvPI = 1.0 / PI;
	constexpr double Eps = std::numeric_limits<double>::epsilon();

} // namespace pls
