#include "stdafx.h"

#include "StitchingWrapper.h"

array<System::Double>^ Stitching::StitchWrapper::CalculateStitchLR(array<array<System::Byte>^>^ scan_row)
{
	array<System::Double>^  vector = gcnew array<System::Double>(scan_row->Length);
	vector[0] = 500;

	return vector;
}