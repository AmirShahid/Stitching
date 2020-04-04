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

///If the Struct seems Strange it is because of Common Language Infrastructure (CLI) addressing for DLL

/// for C++ use only you won't need these.

struct ImageFile
{
public:
	uchar* data_;
	int length_;
    ImageFile()
    {
		length_ = 0;
		data_ = nullptr;
    }
	ImageFile(const std::vector<uchar>& data, int length)
	{
		data_ = new uchar[length];
		std::copy(data.begin(), data.end(), data_);
		length_ = length;
		
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
	struct ImageFile image_file_;
	int x_,y_,z_;

    FullLamelImage()
    {
		x_ = 0;
		y_ = 0;
		z_ = 0;
    }

    FullLamelImage(ImageFile image_file,int x, int y, int z)
    {
		image_file_ = image_file;
		x_ = x;
		y_ = y;
		z_ = z;
    }
};

struct FullLamelImages
{
public :
	FullLamelImage* full_lamel_image_;
	int length_;
    FullLamelImages(std::vector<FullLamelImage> full_lamel_images)
    {
		full_lamel_image_ = new FullLamelImage[full_lamel_images.size()];
		std::copy(full_lamel_images.begin(), full_lamel_images.end(), full_lamel_image_);
        length_ = full_lamel_images.size();
    }
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
	Stitch(int row_count = 0, int column_count = 0);

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
	/* @brief Calculate horizontal shifts multi threaded in rows
       @return 2d shifts vector
    */
    std::vector<std::vector<shift>> calculate_stitch_shifts_lr_in_row();

    std::vector<shift>& calculate_stitch_shifts_ud();

    /// Calculate horizontal shifts multi threaded in columns
    std::vector<std::vector<shift>> calculate_stitch_shifts_lr_in_column();

    /// Calculate horizontal shifts for specific row multi threaded in rows
    std::vector<shift>& calculate_stitch_shifts_lr(int row_number = 0);

    /// only should be called when horizontal shifts has been calculated
    void Stitch_all();

    const int row_count, column_count;
	std::vector<FullLamelImage> full_lamel_images;
	std::vector<std::vector<double>> stitch_shifts_lr_row;
	std::vector<std::vector<double>> stitch_shifts_lr_col;
	std::vector<int>most_informative_column;
	ImageFile* row_images;
	LamelImages* lamel_images;
	best_column best_column_row;
	std::vector<std::vector<shift>> stitch_shifts_lr;
	

    private:
	std::vector<shift> stitch_shifts_row;
	std::vector<shift> stitch_shifts_ud;
	std::vector<best_column>best_column_for_ud;
	std::vector<std::vector<int>> start_tile_r;
	std::vector<std::vector<int>> start_tile_c;
	std::vector<int> shift_idx;

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
	cv::Mat stitch_and_blend(int start_row, int end_row, int start_col, int end_col, int big_tile_size, int left_margin, int top_margin, const cv::Mat& illumination_pattern);
	cv::Mat get_illumination_pattern();
	blank_property is_image_blank(const cv::Mat& image, int threshold);

	int vertical_deviation = 2;
	int number_of_threads = 4;
	int sample_width = 2448;
	int sample_height = 2048;
	int tile_size = 256;

    // init Parameters
    std::string data_dir = "E:\\lamel_stitching\\whole_lamel_data_5\\";
	std::string image_ext = "jpeg";
	std::string pref = "img_";
    
	bool nogui = true;
	bool load_shift_arrays = true;
	bool store_locally = true;
	bool show_log = false;
	bool load_from_disk = true;
	int zoom_levels = 6;
	bool illumination_pattern_from_hard = false;

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