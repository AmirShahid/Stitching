#pragma once
#include "ManagedObject.h"
#include "../Stitching/Core.h"
#include <vector>

using namespace System;
namespace CLI
{
public ref class Stitch: public ManagedObject<core::Stitch>
{
public:
	Stitch(int row_count, int column_count);
	/*ref struct shift
	{
		double shift_r;
		double shift_c;

	    shift()
        {
			shift_c = INVALID_VALUE;
			shift_r = INVALID_VALUE;
        }
		shift(const double& r, const double& c)
		{
			shift_r = r;
			shift_c = c;
		}*/
	//};
	std::vector<core::Stitch::shift> calculate_stitch_shifts_ud();
	std::vector<std::vector<core::Stitch::shift>> calculate_stitch_shifts_lr_in_column();
};
}