#pragma once
//#include "../Stitching/Stitch.h"
#include <string>

using namespace System;

namespace Stitching {
	public ref class StitchWrapper
	{
	public:
		//StitchWrapper() = default;
		array<System::Double>^ CalculateStitchLR(array<array<System::Byte>^>^ scan_row);
		//void CalculateStitchUD();

	//private:
		//Stitch* stitch;
	};

	public ref class StitchConfig
	{
	public:
		int vertical_deviation;
		int number_of_threads;
		System::String^ data_dir = "E:\\lamel_stitching";
		System::String^ dataset_name = "whole_lamel_data_5";
		System::String^ path_delim = "\\";
		System::String^ image_ext = "jpeg";
		System::String^ pref = "img_";
		// Stitching Parameters
		int split_ratio_lr;
		int split_ratio_ud;
		int maxHighCorrPoint_lr;
		int maxHighCorrPoint_ud;
		float acceptanceThresh_lr;
		float acceptanceThresh_ud;
		float lu_image_portion;
		float expected_shift_r_ud, expected_shift_c_ud,
			max_shift_c_threshold_ud, max_shift_r_threshold_ud;
		float expected_shift_r_lr, expected_shift_c_lr,
			max_shift_c_threshold_lr, max_shift_r_threshold_lr;
		double grey_std_thresh_ST;
		double grey_std_thresh_BC;
		double area_threshold;

	};
}
