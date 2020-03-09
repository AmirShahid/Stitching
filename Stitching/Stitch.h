#pragma once
#include <string>
#include <opencv2/core/mat.hpp>
#include <xtensor/xarray.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace pt = boost::property_tree;

const std::string CONFIG_FILE_PATH = "StitchConfig.json";

#define INVALID_VALUE -50000.0

struct ImageFile
{
public:
	uchar* data;
	int length;
    ImageFile()
    {
		length = 0;
		data = nullptr;
    }
	ImageFile(uchar* data_, int length_)
	{
		length = length_;
		data = data_;
	}
};

struct ImageRow
{
public:
	struct ImageFile* columns;
	int column_count;
};

struct ShiftArrayRow
{
public:
	double* columns;
	int column_count;
};

struct ShiftArray
{
public:
	ShiftArrayRow* rows;
	int row_count;
};

struct LamelImages
{
public:
	struct ImageRow* rows;
	int row_count;
};

struct FullLamelImage
{
public:
	struct ImageFile* image_file;
	int x,y,z;

    FullLamelImage(ImageFile* image_file_,int x_, int y_, int z_)
    {
		image_file = image_file_;
		x = x_;
		y = y_;
		z = z_;
    }
};

struct FullLamelImages
{
public :
	FullLamelImage * full_lamel_image;
	int length;
};

struct FullLamelLevels
{
public:
	struct LamelImages* lamel_images;
	int zoom_level_count;
};

static int instance_count = 0;

class Stitch
{
public:
	Stitch(int row_count = 0, int column_count = 0, int zoom_levels = 0);

    struct best_column
	{
		best_column() { column = 0; color_ratio = 0; }
		best_column(const int column_, const double color_ratio_) { column = column_; color_ratio = color_ratio_; }
		int column;
		double color_ratio;
	};

    struct shift
	{
		double shift_r;
		double shift_c;

		shift() { shift_r = INVALID_VALUE; shift_c = INVALID_VALUE; }

        shift(const double& r,const double& c)
        {
			shift_r = r;
			shift_c = c;
        }
	};
	
	//std::vector<std::vector<shift>> calculate_stitch_shifts_lr_in_row();
	std::vector<shift>& calculate_stitch_shifts_ud();
	//std::vector<std::vector<shift>> calculate_stitch_shifts_lr_in_column();
	//void stitch_single_thread_lr(int start_row, int end_row, int start_col, int end_col);
	std::vector<shift>& calculate_stitch_shifts_lr(int row_number = 0);
	void Stitch_all();
	std::vector<std::vector<shift>> stitch_shifts_lr;
	std::vector<shift> stitch_shifts_row;
	std::vector<shift> stitch_shifts_ud;
	std::vector<std::vector<double>> stitch_shifts_lr_row;
	std::vector<std::vector<double>> stitch_shifts_lr_col;
	std::vector<double> stitch_shifts_ud_row;
	std::vector<double> stitch_shifts_ud_col;
	std::vector<best_column>best_column_for_ud;
	std::vector<int>most_informative_column;
	//std::vector<std::vector<std::vector<ImageFile>>>stitched_images;
	std::vector<std::vector<int>> start_tile_r;
	std::vector<std::vector<int>> start_tile_c;
	best_column best_column_row;
	std::vector<int> shift_idx;
	ImageFile* row_images;
	LamelImages* lamel_images;
	std::vector<FullLamelImage> full_lamel_images;
    struct blank_property
    {
		bool is_blank;
		double color_ratio;
    };
	
	shift stitch_ud(cv::Mat& image_up, cv::Mat& image_down);
	shift stitch_lr(cv::Mat& image_left, cv::Mat& image_right);
	void stitch_single_thread_lr(int start_row, int end_row, int start_col, int end_col);
	void stitch_single_thread_ud(int start_row, int end_row);
	void blend(cv::Mat& crop_stitch_image_mask, cv::Mat& stitched_image, cv::Mat& image, cv::Rect image_rect);
	cv::Mat stitch_and_blend(int start_row, int end_row, int start_col, int end_col, int big_tile_size, int left_margin, int top_margin);

	blank_property is_image_blank(const cv::Mat& image, int threshold);
	
	int vertical_deviation = 2;
	int number_of_threads = 4;
	int sample_width = 2448;
	int sample_height = 2048;
	int zoom_levels, tile_size = 256;
	const int row_count, column_count;
	// init Parameters
    std::string data_dir = "E:\\lamel_stitching";
    std::string dataset_name = "whole_lamel_data_5";
	std::string path_delim = "\\";
	std::string image_ext = "jpeg";
	std::string pref = "img_";
	// Stitching Parameters
	int split_ratio_lr = 6;
	int split_ratio_ud = 6;
	int maxHighCorrPoint_lr = 3;
	int maxHighCorrPoint_ud = 3;
	float acceptanceThresh_lr = 0.95;
	float acceptanceThresh_ud = 0.95;
	float lu_image_portion = 0.7;
	float expected_shift_r_ud = 1536, expected_shift_c_ud = 0,
		max_shift_c_threshold_ud = 2448, max_shift_r_threshold_ud=308;
	float expected_shift_r_lr = 0.0, expected_shift_c_lr = 1836,
		max_shift_c_threshold_lr = 368, max_shift_r_threshold_lr = 308;
	double grey_std_thresh_ST = 11;
	double grey_std_thresh_BC = 5;
	double area_threshold = 0.0001;
	pt::ptree config;
};
extern "C" __declspec(dllexport) int __cdecl get_lr_shifts(ImageRow* row_images, double* shift_r, double* shift_c);
extern "C" __declspec(dllexport) FullLamelImages __cdecl stitch_all(LamelImages* lamel_images, int* best_col, ShiftArray* shift_r, ShiftArray* shift_c);