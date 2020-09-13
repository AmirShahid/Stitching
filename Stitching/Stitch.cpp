#include "Stitch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <conio.h>
#include <ctime>
#include <string> 
#include <cmath>
#include <boost/filesystem.hpp>
#include <cstdio>
#include <cstdlib>
//#include <math.h>
#include <future>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;
namespace pt = boost::property_tree;

Stitch::Stitch(int row_count, int column_count) :
	stitch_shifts_lr(row_count, vector<shift>(column_count,shift())),
	stitch_shifts_lr_row(row_count, vector<double>(column_count, double())),
	stitch_shifts_lr_col(row_count, vector<double>(column_count, double())),
	stitch_shifts_ud(row_count, shift()),
	stitch_shifts_row(column_count),
	best_column_for_ud(row_count, best_column()),
	most_informative_column(row_count),
	start_tile_r(row_count, vector<int>(column_count, 0)),
	start_tile_c(row_count, vector<int>(column_count, 0)),
	row_count(row_count),
	column_count(column_count)
{
	for (int i = -1 * vertical_deviation; i <= vertical_deviation; i++) shift_idx.push_back(i);
	pt::read_json(CONFIG_FILE_PATH, config);
	vertical_deviation = config.get<int>("vertical_deviation");
	number_of_threads = config.get<int>("number_of_threads");
	data_dir = config.get<string>("data_dir");
	image_ext = config.get<string>("image_extension");
	pref = config.get<string>("pref");
	split_ratio_lr = config.get<int>("split_ratio_lr");
	split_ratio_ud = config.get<int>("split_ratio_ud");
	maxHighCorrPoint_lr = config.get<int>("max_high_correlation_points_lr");
	maxHighCorrPoint_ud = config.get<int>("max_high_correlation_points_ud");
	acceptanceThresh_lr = config.get<float>("acceptance_threshold_lr");
	acceptanceThresh_ud = config.get<float>("acceptance_threshold_ud");
	lu_image_portion = config.get<float>("lu_image_portion");
	expected_shift_r_ud = config.get<float>("expected_shift_r_ud");
	expected_shift_c_ud = config.get<float>("expected_shift_c_ud");
	max_shift_c_threshold_ud = config.get<float>("max_shift_c_threshold_ud");
	max_shift_r_threshold_ud = config.get<float>("max_shift_r_threshold_ud");
	expected_shift_r_lr = config.get<float>("expected_shift_r_lr");
	expected_shift_c_lr = config.get<float>("expected_shift_c_lr");
	max_shift_c_threshold_lr = config.get<float>("max_shift_c_threshold_lr");
	max_shift_r_threshold_lr = config.get<float>("max_shift_r_threshold_lr");
	grey_std_thresh_ST = config.get<double>("grey_std_thresh_ST");
	grey_std_thresh_BC = config.get<double>("grey_std_thresh_BC");
	area_threshold = config.get<double>("area_threshold");
	zoom_levels = config.get<int>("zoom_levels");
	tile_size = config.get<int>("tile_size");
	nogui = config.get<bool>("nogui");
	load_from_disk = config.get<bool>("load_from_disk");
	load_shift_arrays = config.get<bool>("load_shift_arrays");
	store_locally = config.get<bool>("store_locally");
	show_log = config.get<bool>("show_log");
	illumination_pattern_from_hard = config.get<bool>("illumination_pattern_from_hard");
	big_tile_size = int(tile_size * pow(2, zoom_levels));
	instance_count++;
}

float round_n(float num, int dec)
{
	double m = (num < 0.0) ? -1.0 : 1.0;   // check if input is negative
	double pwr = pow(10, dec);
	return float(floor((double)num * m * pwr + 0.5) / pwr) * m;
}

bool shift_cleaner(vector<Stitch::shift>& shifts) {
	vector<double> row;
	vector<double> col;
	for (auto value : shifts)  if (value.shift_r != INVALID_VALUE && value.shift_r != NO_DETECTION_VALUE  && value.shift_r != BLANK_VALUE)
	{
		row.push_back(value.shift_r);
		col.push_back(value.shift_c);
	}

	int size = row.size();
	if (size == 0)
	{
		return false;
	}

	std::nth_element(row.begin(), row.begin() + row.size() / 2, row.end());
	std::nth_element(col.begin(), col.begin() + col.size() / 2, col.end());

	for (auto& shift : shifts)
	{
		if (shift.shift_r == INVALID_VALUE || shift.shift_r == NO_DETECTION_VALUE || shift.shift_r == BLANK_VALUE)
			shift.shift_r = row[row.size() / 2];
		if (shift.shift_c == INVALID_VALUE || shift.shift_c == NO_DETECTION_VALUE || shift.shift_c == BLANK_VALUE)
			shift.shift_c = col[col.size() / 2];
	}
	return true;
}

double calc_median(vector<double> shifts)
{
	vector<double> new_shifts;
	size_t size = shifts.size();
	for (double v : shifts) if (v != INVALID_VALUE && v != BLANK_VALUE && v != NO_DETECTION_VALUE) new_shifts.push_back(v);

	sort(new_shifts.begin(), new_shifts.end());
	size = new_shifts.size();
	if (size == 0)
	{
		return INVALID_VALUE;
	}
	if (size % 2 == 0)
	{
		return (new_shifts[size / 2 - 1] + new_shifts[size / 2]) / 2;
	}
	else
	{
		return new_shifts[size / 2];
	}
}

Stitch::blank_property Stitch::is_image_blank(const cv::Mat& image, int grey_threshold)
{
	int col_num = image.cols, row_num = image.rows;
	Mat BGR_layer[3];
	Mat absdiff_bg, absdiff_gr, absdiff_br, colorness_mat;
	split(image, BGR_layer);

	absdiff(BGR_layer[0], BGR_layer[1], absdiff_bg);
	absdiff(BGR_layer[1], BGR_layer[2], absdiff_gr);
	absdiff(BGR_layer[0], BGR_layer[2], absdiff_br);

	threshold(absdiff_bg, absdiff_bg, 2 * grey_threshold, 255.0, THRESH_BINARY);
	threshold(absdiff_gr, absdiff_gr, 2 * grey_threshold, 255.0, THRESH_BINARY);
	threshold(absdiff_br, absdiff_br, 2 * grey_threshold, 255.0, THRESH_BINARY);

	bitwise_or(absdiff_bg, absdiff_gr, colorness_mat);
	bitwise_or(absdiff_br, colorness_mat, colorness_mat);

	const double color_ratio = sum(colorness_mat)[0] / (image.cols * image.rows) / 255.0;

	blank_property result;

	result.color_ratio = color_ratio;
	if (color_ratio < area_threshold)
	{
		result.is_blank = true;
		return result;
	};
	result.is_blank = false;
	return result;
}

Stitch::shift Stitch::stitch_ud(Mat& image_up, Mat& image_down)
{
	if (image_up.empty() || image_down.empty())
		return shift(INVALID_VALUE, INVALID_VALUE);

	shift stitch_shift;
	Mat image_cur_split, corrMatrix, image_up_gray, image_up_gray_cropped, image_down_gray;
	double max_value_corr, min_value_corr;
	int center_c;
	Point min_loc_corr, max_loc_corr, match_loc;
	cvtColor(image_up, image_up_gray, COLOR_BGR2GRAY);
	cvtColor(image_down, image_down_gray, COLOR_BGR2GRAY);
	int effective_y = (int)(image_up_gray.rows * lu_image_portion);
	image_up_gray(Rect(0, effective_y, image_up_gray.cols, (int)(image_up_gray.rows - effective_y))).copyTo(image_up_gray_cropped);

	vector<double> shift_direction_r;
	vector<double> shift_direction_c;

	int step_r = (int)(image_down.rows / split_ratio_ud);
	int step_c = (int)(image_down.cols / split_ratio_ud);

	// Check Blankness

	blank_property blank_stat = is_image_blank(image_down(Rect(0, 0, image_down.cols, step_r)), grey_std_thresh_ST);
	if (blank_stat.is_blank)
	{
		return shift(BLANK_VALUE, BLANK_VALUE);
	}
	else
	{
		cv::Range range[2];
		for (int j = 0; j < split_ratio_ud; j++)
		{
			if (shift_direction_c.size() >= maxHighCorrPoint_ud) break;

			range[0] = Range(0, (long)step_r);
			range[1] = Range((long)(step_c * j), (long)(step_c * (j + 1)));
			image_cur_split = image_down_gray(range);

			matchTemplate(image_up_gray_cropped, image_cur_split, corrMatrix, TM_CCOEFF_NORMED);
			minMaxLoc(corrMatrix, &min_value_corr, &max_value_corr, &min_loc_corr, &max_loc_corr, Mat());
			if (max_value_corr > acceptanceThresh_ud)
			{
				match_loc = max_loc_corr;
				match_loc.y += effective_y;
				center_c = (long)(image_down.cols / split_ratio_ud * j);
				if (((float)abs((match_loc.x - center_c) - expected_shift_c_ud)) > max_shift_c_threshold_ud ||
					((float)abs((match_loc.y - expected_shift_r_ud))) > max_shift_r_threshold_ud) continue;

				shift_direction_r.push_back((double)(match_loc.y));
				shift_direction_c.push_back((double)(match_loc.x - center_c));

			}
		}
		if (shift_direction_c.empty())
		{
			return shift(NO_DETECTION_VALUE, NO_DETECTION_VALUE);
		}
		else
		{
			stitch_shift.shift_c = (double)(calc_median(shift_direction_c));
			stitch_shift.shift_r = (double)(calc_median(shift_direction_r));
			return stitch_shift;
		}
	}
}

Stitch::shift Stitch::stitch_lr(Mat& image_left, Mat& image_right) {

	if (image_left.empty() || image_right.empty())
		return shift(INVALID_VALUE, INVALID_VALUE);
	shift stitch_shift;
	double max_value_corr, min_value_corr;
	int center_r;
	Point min_loc_corr, max_loc_corr, match_loc;
	Mat image_left_gray, image_left_gray_cropped, image_right_gray, image_cur_split, corrMatrix;
	cvtColor(image_left, image_left_gray, COLOR_BGR2GRAY);
	cvtColor(image_right, image_right_gray, COLOR_BGR2GRAY);
	int effective_x = (int)(image_left_gray.cols * lu_image_portion);
	image_left_gray(Rect(effective_x, 0, (int)(image_left_gray.cols - effective_x), image_left_gray.rows)).copyTo(image_left_gray_cropped);

	// init list and parameters
	vector<double> shift_direction_r;
	vector<double> shift_direction_c;
	int temp_r = (int)(image_right_gray.rows / split_ratio_lr);
	int temp_c = (int)(image_right_gray.cols / split_ratio_lr);

	// Check Blankness
	blank_property blank_stat = is_image_blank(image_right(Rect(0, 0, temp_c, image_right.rows)), grey_std_thresh_ST);

	if (blank_stat.is_blank)
	{
		return shift(BLANK_VALUE,BLANK_VALUE);
	}
	else
	{
		cv::Range range[2];

		for (int i = 0; i < split_ratio_lr && shift_direction_c.size() < maxHighCorrPoint_lr; i++)
		{

			range[0] = Range((long)(temp_r * i), (long)(temp_r * (i + 1)));
			range[1] = Range(0, (long)(temp_c));
			image_cur_split = image_right_gray(range);

			matchTemplate(image_left_gray_cropped, image_cur_split, corrMatrix, TM_CCOEFF_NORMED);
			minMaxLoc(corrMatrix, &min_value_corr, &max_value_corr, &min_loc_corr, &max_loc_corr, Mat());
			if (max_value_corr > acceptanceThresh_lr)
			{
				match_loc = max_loc_corr;
				match_loc.x += effective_x;

				center_r = (long)(image_right_gray.rows / split_ratio_lr * i);

				//Check unexpected shift
				if (abs(match_loc.x - expected_shift_c_lr) > max_shift_c_threshold_lr || \
					abs((match_loc.y - center_r) - expected_shift_r_lr) > max_shift_r_threshold_lr) continue;

				shift_direction_r.push_back((double)(match_loc.y - center_r));
				shift_direction_c.push_back((double)(match_loc.x));
			}

		}
		if (shift_direction_c.empty())
		{
			return shift(NO_DETECTION_VALUE, NO_DETECTION_VALUE);
		}
		else
		{
			stitch_shift.shift_c = (double)(calc_median(shift_direction_c));
			stitch_shift.shift_r = (double)(calc_median(shift_direction_r));
			return stitch_shift;
		}
	}
}

void Stitch::stitch_single_thread_ud(int start_row, int end_row)
{
	Mat image_up, image_down;
	vector<double> temp_shift_r(2 * vertical_deviation + 1, 0.0);
	vector<double> temp_shift_c(2 * vertical_deviation + 1, 0.0);

	for (auto i = start_row; i < end_row; i++)
	{
		if (load_from_disk)
			image_up = imread(data_dir + pref + to_string(i) + "_" + to_string(most_informative_column[i]) + "." + image_ext, IMREAD_UNCHANGED);
		else
			image_up = imdecode(Mat(1, lamel_images->rows[i].columns[most_informative_column[i]].length_, CV_8UC1, &lamel_images->rows[i].columns[most_informative_column[i]].data_[0]), IMREAD_UNCHANGED);
		for (auto H_shift : shift_idx)
		{
			if (load_from_disk)
				image_down = imread(data_dir + pref + to_string(i + 1) + "_" +
					to_string(most_informative_column[i] + H_shift) + "." + image_ext, IMREAD_UNCHANGED);
			else
				image_down = imdecode(Mat(1, lamel_images->rows[i + 1].columns[most_informative_column[i]].length_, CV_8UC1, &lamel_images->rows[i + 1].columns[most_informative_column[i]].data_[0]), IMREAD_UNCHANGED);
			shift stitch_result = stitch_ud(image_up, image_down);
			if (show_log)
			{
				if (stitch_result.shift_r == NO_DETECTION_VALUE)
					cout << "Row: " << i << " Col: " << most_informative_column[i] + H_shift << endl << "Result: No Detection" << endl;
				else if (stitch_result.shift_r == BLANK_VALUE)
					cout << "Row: " << i << " Col: " << most_informative_column[i] + H_shift << endl << "Result: BLANK" << endl;
				else if (stitch_result.shift_r == INVALID_VALUE)
					cout << "Row: " << i << " Col: " << most_informative_column[i] + H_shift << endl << "Result: INVALID" << endl;
				else
					cout << "Row: " << i << " Col: " << most_informative_column[i] + H_shift << endl << "Result: " << stitch_result.shift_r << ", " << stitch_result.shift_c << endl;
			}
			temp_shift_r[H_shift + vertical_deviation] = stitch_result.shift_r;
			temp_shift_c[H_shift + vertical_deviation] = stitch_result.shift_c;

			if (temp_shift_c[H_shift + vertical_deviation] != INVALID_VALUE && temp_shift_c[H_shift + vertical_deviation] != NO_DETECTION_VALUE && temp_shift_c[H_shift + vertical_deviation] != BLANK_VALUE)
			{
				if (H_shift < 0)
				{

					for (int k = H_shift + most_informative_column[i]; k < most_informative_column[i]; k++)
					{
						if (stitch_shifts_lr_row[i][k] != INVALID_VALUE && stitch_shifts_lr_row[i][k] != NO_DETECTION_VALUE && stitch_shifts_lr_row[i][k] != BLANK_VALUE)
							temp_shift_r[H_shift + vertical_deviation] += stitch_shifts_lr_row[i][k];
						if (stitch_shifts_lr_col[i][k] != INVALID_VALUE && stitch_shifts_lr_col[i][k] != BLANK_VALUE && stitch_shifts_lr_col[i][k] != NO_DETECTION_VALUE)
							temp_shift_c[H_shift + vertical_deviation] += stitch_shifts_lr_col[i][k];
					}
				}
				else if (H_shift > 0)
				{
					for (int k = H_shift + most_informative_column[i]; k > most_informative_column[i]; k--)
					{
						if (stitch_shifts_lr_row[i][k] != INVALID_VALUE && stitch_shifts_lr_row[i][k] != NO_DETECTION_VALUE && stitch_shifts_lr_row[i][k] != BLANK_VALUE)
							temp_shift_r[H_shift + vertical_deviation] -= stitch_shifts_lr_row[i][k];
						if (stitch_shifts_lr_col[i][k] != INVALID_VALUE && stitch_shifts_lr_col[i][k] != BLANK_VALUE && stitch_shifts_lr_col[i][k] != NO_DETECTION_VALUE)
							temp_shift_c[H_shift + vertical_deviation] -= stitch_shifts_lr_col[i][k];
					}
				}
			}
		}
		const double median_r = calc_median(temp_shift_r);
		const double median_c = calc_median(temp_shift_c);
		stitch_shifts_ud[i] = shift(median_r, median_c);

	}
}

void Stitch::stitch_single_thread_lr(int start_row, int end_row, int start_col, int end_col)
{
	Mat image_left, image_right;
	blank_property blank_stat;
	for (auto i = start_row; i <= end_row; i++)
	{
		for (auto j = start_col; j < end_col; j++)
		{
			if (!image_right.empty())
				image_left = image_right;
			else
			{
				if (load_from_disk)
				{
					image_left = imread(data_dir + pref + to_string(i) + "_" + to_string(j) + "." + image_ext);
				}
				else
					image_left = imdecode(Mat(1, row_images[j].length_, CV_8UC1, &row_images[j].data_[0]), IMREAD_UNCHANGED);
			}
			if (load_from_disk)
				image_right = imread(data_dir + pref + to_string(i) + "_" + to_string(j + 1) + "." + image_ext);
			else
				image_right = imdecode(Mat(1, row_images[j + 1].length_, CV_8UC1, &row_images[j + 1].data_[0]), IMREAD_UNCHANGED);
		    stitch_shifts_row[j] = stitch_lr(image_left, image_right);
			blank_stat = is_image_blank(image_left(Rect(0, (int)(lu_image_portion * image_left.rows), image_left.cols, image_left.rows - (int)(lu_image_portion * image_left.rows))), grey_std_thresh_BC);
			if (show_log)
			{
				if (stitch_shifts_row[j].shift_r == NO_DETECTION_VALUE)
					cout << "Row: " << i << " Col: " << j << endl << "Result: No Detection" << endl << "Color Ratio: " << blank_stat.color_ratio << endl;
				else if (stitch_shifts_row[j].shift_r == BLANK_VALUE)
					cout << "Row: " << i << " Col: " << j << endl << "Result: BLANK" << endl << "Color Ratio: " << blank_stat.color_ratio << endl;
				else if (stitch_shifts_row[j].shift_r == INVALID_VALUE)
					cout << "Row: " << i << " Col: " << j << endl << "Result: INVALID" << endl << "Color Ratio: " << blank_stat.color_ratio << endl;
				else
					cout << "Row: " << i << " Col: " << j << endl << "Result: " << stitch_shifts_row[j].shift_r << ", " << stitch_shifts_row[j].shift_c << endl << "Color Ratio: " << blank_stat.color_ratio << endl;
			}
			if (best_column_row.color_ratio < blank_stat.color_ratio)
				best_column_row = best_column(j, blank_stat.color_ratio);
		}
	}
}

vector<Stitch::shift>& Stitch::calculate_stitch_shifts_lr(int row_number)
{
	vector<future<void>> lr_threads;
	int offset = column_count / number_of_threads;
    for (int j = 0; j < column_count; j = j + offset)
	{
		lr_threads.push_back(async(launch::async, &Stitch::stitch_single_thread_lr, this, row_number, row_number, j, min((j + offset + 1), column_count - 1)));
	}

	for (auto &thread : lr_threads)
	{
		thread.wait();
	}
	lr_threads.clear();
	bool is_succesful = shift_cleaner(stitch_shifts_row);
	if (!is_succesful)
		throw Exception(-50000, "Failed to calculate LR shift", "calculate shift_lr", "stitch.cpp", 422);

	//for (int j = 0; j < stitch_shifts_lr.size(); j++) 
	//{
	//	stitch_shifts_lr_row[row_number][j] = stitch_shifts_lr[j].shift_r;
	//	stitch_shifts_lr_col[row_number][j] = stitch_shifts_lr[j].shift_c;
	//}
	//most_informative_column[row_number] = best_column_row.column;

	if (store_locally) {
		cout << "saving shift arrays" << endl;
		pt::ptree json_arr;
		for (int i = 0; i < stitch_shifts_row.size(); i++)
		{
			pt::ptree shift_pair;
			shift_pair.put("shift_r", stitch_shifts_row[i].shift_r);
			shift_pair.put("shift_c", stitch_shifts_row[i].shift_c);
			json_arr.push_back(make_pair(to_string(i), shift_pair));
		}
		pt::ptree root;
		root.add_child(to_string(row_number), json_arr);
		pt::ptree best_col;
		best_col.put("column", best_column_row.column);
		best_col.put("color_ratio", best_column_row.color_ratio);
		root.add_child("most_informative_column", best_col);
		pt::write_json("LR_" + to_string(row_number) + ".json", root);
	}
	return stitch_shifts_row;
}

vector<Stitch::shift>& Stitch::calculate_stitch_shifts_ud()
{
	if (load_shift_arrays)
	{
		cout << "reading shift arrays" << endl;
		pt::ptree lrS;
		for (int i = 0; i < row_count; i++)
		{
			cout << "lr_ " + to_string(i) + " shifts read from json" << endl;
			pt::read_json("LR_" + to_string(i) + ".json", lrS);
			for (int j = 0; j < column_count; j++)
			{
				stitch_shifts_lr_row[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_r");
				stitch_shifts_lr_col[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_c");
			}
			most_informative_column[i] = lrS.get<int>("most_informative_column.column");
		}
	}
	vector<future<void>> ud_threads;
	int offset = row_count / number_of_threads;
	for (int j = 0; j < row_count; j = j + offset)
	{
		cout << "start thread" << endl;
		ud_threads.push_back(async(launch::async, &Stitch::stitch_single_thread_ud, this, j, min(j + offset + 1, row_count - 1)));
	}
	for (auto& thread : ud_threads)
	{
		cout << "waiting for thread" << endl;
		thread.wait();
	}
	cout << "all threads end" << endl;
	ud_threads.clear();
	bool is_succesful = shift_cleaner(stitch_shifts_ud);
	if (!is_succesful)
		throw Exception(-50000, "Failed to calculate UD shift", "calculate shift_ud", "stitch.cpp", 486);

	if (store_locally)
	{
		cout << "saving shift arrays" << endl;
		pt::ptree json_arr;
		for (int i = 0; i < stitch_shifts_ud.size(); i++)
		{
			pt::ptree shift_pair;
			shift_pair.put("shift_r", stitch_shifts_ud[i].shift_r);
			shift_pair.put("shift_c", stitch_shifts_ud[i].shift_c);
			json_arr.push_back(make_pair(to_string(i), shift_pair));
		}
		pt::ptree root;
		root.add_child("UD", json_arr);
		pt::write_json("UD.json", root);
	}
	return stitch_shifts_ud;
}

void update_min_and_max(int new_value, int& min_val, int& max_val) {
	if (new_value > max_val)
		max_val = new_value;
	if (new_value < min_val)
		min_val = new_value;
}

vector<vector<vector<int>>> Stitch::get_big_tile_coordinates()
{
	pt::ptree lrS;
	pt::ptree udS;
	if (load_shift_arrays)
	{
		for (int i = 0; i < row_count; i++)
		{
			pt::read_json("LR_" + to_string(i) + ".json", lrS);
			pt::read_json("UD.json", udS);
			udS = udS.get_child("UD");
			for (int j = 0; j < column_count; j++)
			{
				stitch_shifts_lr_row[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_r");
				stitch_shifts_lr_col[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_c");
			}
			stitch_shifts_ud[i].shift_r = udS.get<double>(to_string(i) + ".shift_r");
			stitch_shifts_ud[i].shift_c = udS.get<double>(to_string(i) + ".shift_c");
			most_informative_column[i] = lrS.get<int>("most_informative_column.column");
		}
	}
	cout << "1" << endl;
	int start_row = 0, end_row = row_count - 1, start_col = 2, end_col = column_count - 1;
	int min_r = 0, max_r = 0, min_c = 0, max_c = 0;

	for (int cc = start_col + 1; cc <= end_col; cc++)
	{
		start_tile_r[start_row][cc] = start_tile_r[start_row][cc - 1]
			+ stitch_shifts_lr_row[start_row][cc - 1];
		start_tile_c[start_row][cc] = start_tile_c[start_row][cc - 1]
			+ stitch_shifts_lr_col[start_row][cc - 1];
		update_min_and_max(start_tile_r[start_row][cc], min_r, max_r);
		update_min_and_max(start_tile_c[start_row][cc], min_c, max_c);
	}
	cout << "2" << endl;
	for (int rr = start_row + 1; rr <= end_row; rr++)
	{
		start_tile_r[rr][most_informative_column[rr - 1]] =
			start_tile_r[rr - 1][most_informative_column[rr - 1]] + stitch_shifts_ud[rr - 1].shift_r;

		start_tile_c[rr][most_informative_column[rr - 1]] =
			start_tile_c[rr - 1][most_informative_column[rr - 1]] + stitch_shifts_ud[rr - 1].shift_c;

		update_min_and_max(start_tile_r[rr][most_informative_column[rr - 1]],
			min_r, max_r);
		update_min_and_max(start_tile_c[rr][most_informative_column[rr - 1]],
			min_c, max_c);

		for (int cc = most_informative_column[rr - 1] + 1; cc <= end_col; cc++)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][most_informative_column[rr - 1]];
			start_tile_c[rr][cc] = start_tile_c[rr][most_informative_column[rr - 1]];
			for (int k = most_informative_column[rr - 1]; k < cc; k++)
			{
				start_tile_r[rr][cc] += stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] += stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}
		cout << "3" << endl;
		for (int cc = most_informative_column[rr - 1] - 1; cc >= start_col; cc--)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][most_informative_column[rr - 1]];
			start_tile_c[rr][cc] = start_tile_c[rr][most_informative_column[rr - 1]];
			for (int k = most_informative_column[rr - 1] - 1; k >= cc; k--)
			{
				cout << "4" << endl;
				start_tile_r[rr][cc] -= stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] -= stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}
	}
	// Calculating output width and height
	int output_width = max_c - min_c + sample_width;
	int output_height = max_r - min_r + sample_height;

	// Shifting min
	for (int rr = start_row; rr <= end_row; rr++)
	{
		for (int cc = start_col; cc <= end_col; cc++)
		{
			cout << "5" << endl;
			start_tile_c[rr][cc] -= min_c;
			start_tile_r[rr][cc] -= min_r;
		}
	}

	cout << "Lamel: output width: " << output_width << " " << "output Height: " << output_height << endl;

	int number_of_big_tile_c = output_width / big_tile_size + 1;
	int number_of_big_tile_r = output_height / big_tile_size + 1;

	vector<vector<vector<int>>> tile_config_array(number_of_big_tile_r, vector<vector<int>>(number_of_big_tile_c, vector<int>(6, INVALID_VALUE)));

	int current_tile_row_index, current_tile_col_index;
	for (int rr = start_row; rr <= end_row; rr++)
	{
		for (int cc = start_col; cc <= end_col; cc++)
		{
			current_tile_col_index = int(start_tile_c[rr][cc] / big_tile_size);
			current_tile_row_index = int(start_tile_r[rr][cc] / big_tile_size);

			// Update start row and top margin
			if (int((start_tile_r[rr][cc] + sample_height) / big_tile_size) > current_tile_row_index)
			{
				if ((rr < tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] &&
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] != INVALID_VALUE) ||
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] == INVALID_VALUE)
				{

					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] = rr;
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][TOP_MARGIN] =
						(current_tile_row_index + 1) * big_tile_size - start_tile_r[rr][cc];
				}

			}
			if ((rr < tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] == INVALID_VALUE)
			{

				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] = rr;
				tile_config_array[current_tile_row_index][current_tile_col_index][TOP_MARGIN] =
					current_tile_row_index * big_tile_size - start_tile_r[rr][cc];
			}

			// Update end row
			if ((rr > tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] == INVALID_VALUE)
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] = rr;

			// Update start column and left margin
			if (int((start_tile_c[rr][cc] + sample_width) / big_tile_size) > current_tile_col_index)
			{
				if ((cc < tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] &&
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] != INVALID_VALUE) ||
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] == INVALID_VALUE)
				{

					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] = cc;
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][LEFT_MARGIN] =
						(current_tile_col_index + 1) * big_tile_size - start_tile_c[rr][cc];
				}

			}
			if ((cc < tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] == INVALID_VALUE)
			{

				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] = cc;
				tile_config_array[current_tile_row_index][current_tile_col_index][LEFT_MARGIN] =
					current_tile_col_index * big_tile_size - start_tile_c[rr][cc];
			}

			// Update end column
			if ((cc > tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] == INVALID_VALUE)
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] = cc;
		}
		cout << "6" << endl;
	}
	return tile_config_array;
}

void Stitch::Stitch_big_tile(int idx_r, int idx_c)
{
	//Mat illumination_pattern = get_illumination_pattern();
	Mat illumination_pattern = Mat();
	Mat current_big_tile;
	int is_empty_zone;
	is_empty_zone = 0;
	if (tile_config_array[idx_r][idx_c][START_ROW] != INVALID_VALUE)
	{
		cout << "BIG_TILE: " << idx_r << " " << idx_c << endl;
		cout << "START_ROW: " << tile_config_array[idx_r][idx_c][START_ROW] << endl;
		cout << "END_ROW: " << tile_config_array[idx_r][idx_c][END_ROW] << endl;
		cout << "START_COL: " << tile_config_array[idx_r][idx_c][START_COL] << endl;
		cout << "END_COL: " << tile_config_array[idx_r][idx_c][END_COL] << endl;
		cout << "LEFT_MARGIN: " << tile_config_array[idx_r][idx_c][LEFT_MARGIN] << endl;
		cout << "TOP_MARGIN: " << tile_config_array[idx_r][idx_c][TOP_MARGIN] << endl;
		current_big_tile = stitch_and_blend(tile_config_array[idx_r][idx_c][START_ROW],
			tile_config_array[idx_r][idx_c][END_ROW],
			tile_config_array[idx_r][idx_c][START_COL],
			tile_config_array[idx_r][idx_c][END_COL],
			big_tile_size, tile_config_array[idx_r][idx_c][LEFT_MARGIN],
			tile_config_array[idx_r][idx_c][TOP_MARGIN], illumination_pattern);
	}
	else
	{
		is_empty_zone = 1;
	}
	int row_bias, col_bias;
	Mat big_tile_clone, current_tile;
	for (int zz = zoom_levels; zz >= 0; zz--)
	{
		row_bias = idx_r * int(pow(2, zz));
		col_bias = idx_c * int(pow(2, zz));
		if (is_empty_zone == 0)
		{
			big_tile_clone = current_big_tile.clone();
		}
		else
		{
			big_tile_clone = Mat::zeros(Size(big_tile_size, big_tile_size), CV_8UC3);
		}
		for (int dd = 0; dd < (zoom_levels - zz); dd++)
		{
			resize(big_tile_clone, big_tile_clone, Size(int(big_tile_clone.cols / 2), int(big_tile_clone.rows / 2)));
		}
		for (int yy = 0; yy<int(big_tile_clone.rows / tile_size); yy++)
		{
			for (int xx = 0; xx<int(big_tile_clone.cols / tile_size); xx++)
			{
				current_tile = big_tile_clone.colRange(xx * tile_size, (xx + 1) * tile_size)
					.rowRange(yy * tile_size, (yy + 1) * tile_size);
				if (!nogui)
				{
					imshow("big tile", current_tile);
					imwrite("output_" + to_string(zoom_levels) + "\\" + to_string(col_bias + xx) + "_" + to_string(row_bias + yy) + "_" + to_string(zz) + ".jpeg", current_tile);
					waitKey(100);
				}
				// Z = zz | Y = row_bias + yy | X = col_bias + xx
				vector<uchar> encoded_image;
				cvtColor(current_tile, current_tile, COLOR_BGR2RGB);
				imencode(".jpeg", current_tile, encoded_image);
				ImageFile image_file(encoded_image, encoded_image.size());
				full_lamel_images.push_back(FullLamelImage(image_file, col_bias + xx, row_bias + yy, zz));
			}
		}
	}
	cout << "SUCCESSFULLY ENDED! ;)" << endl;
}

void Stitch::Stitch_all()
{
	pt::ptree lrS;
	pt::ptree udS;
	if (load_shift_arrays)
	{
		for (int i = 0; i < row_count; i++)
		{
			pt::read_json("LR_" + to_string(i) + ".json", lrS);
			pt::read_json("UD.json", udS);
			udS = udS.get_child("UD");
			for (int j = 0; j < column_count; j++)
			{
				stitch_shifts_lr_row[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_r");
				stitch_shifts_lr_col[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_c");
			}
			stitch_shifts_ud[i].shift_r = udS.get<double>(to_string(i) + ".shift_r");
			stitch_shifts_ud[i].shift_c = udS.get<double>(to_string(i) + ".shift_c");
			most_informative_column[i] = lrS.get<int>("most_informative_column.column");
		}
	}

	int start_row = 0, end_row = row_count - 1, start_col = 2, end_col = column_count - 1;
	int min_r = 0, max_r = 0, min_c = 0, max_c = 0;

	for (int cc = start_col + 1; cc <= end_col; cc++)
	{
		start_tile_r[start_row][cc] = start_tile_r[start_row][cc - 1]
			+ stitch_shifts_lr_row[start_row][cc - 1];
		start_tile_c[start_row][cc] = start_tile_c[start_row][cc - 1]
			+ stitch_shifts_lr_col[start_row][cc - 1];
		update_min_and_max(start_tile_r[start_row][cc], min_r, max_r);
		update_min_and_max(start_tile_c[start_row][cc], min_c, max_c);
	}

	for (int rr = start_row + 1; rr <= end_row; rr++)
	{
		start_tile_r[rr][most_informative_column[rr - 1]] =
			start_tile_r[rr - 1][most_informative_column[rr - 1]] + stitch_shifts_ud[rr - 1].shift_r;

		start_tile_c[rr][most_informative_column[rr - 1]] =
			start_tile_c[rr - 1][most_informative_column[rr - 1]] + stitch_shifts_ud[rr - 1].shift_c;

		update_min_and_max(start_tile_r[rr][most_informative_column[rr - 1]],
			min_r, max_r);
		update_min_and_max(start_tile_c[rr][most_informative_column[rr - 1]],
			min_c, max_c);

		for (int cc = most_informative_column[rr - 1] + 1; cc <= end_col; cc++)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][most_informative_column[rr - 1]];
			start_tile_c[rr][cc] = start_tile_c[rr][most_informative_column[rr - 1]];
			for (int k = most_informative_column[rr - 1]; k < cc; k++)
			{
				start_tile_r[rr][cc] += stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] += stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}

		for (int cc = most_informative_column[rr - 1] - 1; cc >= start_col; cc--)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][most_informative_column[rr - 1]];
			start_tile_c[rr][cc] = start_tile_c[rr][most_informative_column[rr - 1]];
			for (int k = most_informative_column[rr - 1] - 1; k >= cc; k--)
			{
				start_tile_r[rr][cc] -= stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] -= stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}
	}

	// Calculating output width and height
	int output_width = max_c - min_c + sample_width;
	int output_height = max_r - min_r + sample_height;

	// Shifting min
	for (int rr = start_row; rr <= end_row; rr++)
	{
		for (int cc = start_col; cc <= end_col; cc++)
		{
			start_tile_c[rr][cc] -= min_c;
			start_tile_r[rr][cc] -= min_r;
		}
	}

	cout << "Lamel: output width: " << output_width << " " << "output Height: " << output_height << endl;

	int number_of_big_tile_c = output_width / big_tile_size + 1;
	int number_of_big_tile_r = output_height / big_tile_size + 1;

	vector<vector<vector<int>>> tile_config_array(number_of_big_tile_r, vector<vector<int>>(number_of_big_tile_c, vector<int>(6, INVALID_VALUE)));

	int current_tile_row_index, current_tile_col_index;
	for (int rr = start_row; rr <= end_row; rr++)
	{
		for (int cc = start_col; cc <= end_col; cc++)
		{
			current_tile_col_index = int(start_tile_c[rr][cc] / big_tile_size);
			current_tile_row_index = int(start_tile_r[rr][cc] / big_tile_size);

			// Update start row and top margin
			if (int((start_tile_r[rr][cc] + sample_height) / big_tile_size) > current_tile_row_index)
			{
				if ((rr < tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] &&
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] != INVALID_VALUE) ||
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] == INVALID_VALUE)
				{

					tile_config_array[current_tile_row_index + 1][current_tile_col_index][START_ROW] = rr;
					tile_config_array[current_tile_row_index + 1][current_tile_col_index][TOP_MARGIN] =
						(current_tile_row_index + 1) * big_tile_size - start_tile_r[rr][cc];
				}

			}
			if ((rr < tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] == INVALID_VALUE)
			{

				tile_config_array[current_tile_row_index][current_tile_col_index][START_ROW] = rr;
				tile_config_array[current_tile_row_index][current_tile_col_index][TOP_MARGIN] =
					start_tile_r[rr][cc] - current_tile_row_index * big_tile_size;
			}

			// Update end row
			if ((rr > tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] == INVALID_VALUE)
				tile_config_array[current_tile_row_index][current_tile_col_index][END_ROW] = rr;

			// Update start column and left margin
			if (int((start_tile_c[rr][cc] + sample_width) / big_tile_size) > current_tile_col_index)
			{
				if ((cc < tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] &&
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] != INVALID_VALUE) ||
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] == INVALID_VALUE)
				{

					tile_config_array[current_tile_row_index][current_tile_col_index + 1][START_COL] = cc;
					tile_config_array[current_tile_row_index][current_tile_col_index + 1][LEFT_MARGIN] =
						(current_tile_col_index + 1) * big_tile_size - start_tile_c[rr][cc];
				}

			}
			if ((cc < tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] == INVALID_VALUE)
			{

				tile_config_array[current_tile_row_index][current_tile_col_index][START_COL] = cc;
				tile_config_array[current_tile_row_index][current_tile_col_index][LEFT_MARGIN] =
					start_tile_c[rr][cc] - current_tile_col_index * big_tile_size;
			}

			// Update end column
			if ((cc > tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] &&
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] != INVALID_VALUE) ||
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] == INVALID_VALUE)
				tile_config_array[current_tile_row_index][current_tile_col_index][END_COL] = cc;
		}
	}

	// Generating tileset
	Mat illumination_pattern = get_illumination_pattern();
	Mat current_big_tile;
	int is_empty_zone;
	for (int idx_r = 0; idx_r < number_of_big_tile_r; idx_r++)
	{
		for (int idx_c = 0; idx_c < number_of_big_tile_c; idx_c++)
		{
			cout << "#################################################" << endl;
			is_empty_zone = 0;
			if (tile_config_array[idx_r][idx_c][START_ROW] != INVALID_VALUE)
			{
				cout << "BIG_TILE: " << idx_r << " " << idx_c << endl;
				cout << "START_ROW: " << tile_config_array[idx_r][idx_c][START_ROW] << endl;
				cout << "END_ROW: " << tile_config_array[idx_r][idx_c][END_ROW] << endl;
				cout << "START_COL: " << tile_config_array[idx_r][idx_c][START_COL] << endl;
				cout << "END_COL: " << tile_config_array[idx_r][idx_c][END_COL] << endl;
				cout << "LEFT_MARGIN: " << tile_config_array[idx_r][idx_c][LEFT_MARGIN] << endl;
				cout << "TOP_MARGIN: " << tile_config_array[idx_r][idx_c][TOP_MARGIN] << endl;
				current_big_tile = stitch_and_blend(tile_config_array[idx_r][idx_c][START_ROW],
					tile_config_array[idx_r][idx_c][END_ROW],
					tile_config_array[idx_r][idx_c][START_COL],
					tile_config_array[idx_r][idx_c][END_COL],
					big_tile_size, tile_config_array[idx_r][idx_c][LEFT_MARGIN],
					tile_config_array[idx_r][idx_c][TOP_MARGIN], illumination_pattern);
			}
			else
			{
				is_empty_zone = 1;
			}
			int row_bias, col_bias;
			Mat big_tile_clone, current_tile;
			for (int zz = zoom_levels; zz >= 0; zz--)
			{
				row_bias = idx_r * int(pow(2, zz));
				col_bias = idx_c * int(pow(2, zz));
				if (is_empty_zone == 0)
				{
					big_tile_clone = current_big_tile.clone();
				}
				else
				{
					big_tile_clone = Mat::zeros(Size(big_tile_size, big_tile_size), CV_8UC3);
				}
				for (int dd = 0; dd < (zoom_levels - zz); dd++)
				{
					resize(big_tile_clone, big_tile_clone, Size(int(big_tile_clone.cols / 2), int(big_tile_clone.rows / 2)));
				}
				for (int yy = 0; yy<int(big_tile_clone.rows / tile_size); yy++)
				{
					for (int xx = 0; xx<int(big_tile_clone.cols / tile_size); xx++)
					{
						current_tile = big_tile_clone.colRange(xx * tile_size, (xx + 1) * tile_size)
							.rowRange(yy * tile_size, (yy + 1) * tile_size);
						if (!nogui)
						{
							imshow("big tile", current_tile);
							imwrite("output_" + to_string(zoom_levels) + "\\" + to_string(col_bias + xx) + "_" + to_string(row_bias + yy) + "_" + to_string(zz) + ".jpeg", current_tile);
							waitKey(100);
						}
						// Z = zz | Y = row_bias + yy | X = col_bias + xx
						vector<uchar> encoded_image;
						imencode(".jpeg", current_tile, encoded_image);
						ImageFile image_file(encoded_image, encoded_image.size());
						full_lamel_images.push_back(FullLamelImage(image_file, col_bias + xx, row_bias + yy, zz));
					}
				}
			}
		}
	}
	cout << "SUCCESSFULLY ENDED! ;)" << endl;
}

void Stitch::blend(Mat& crop_stitch_image_mask, Mat& stitched_image, Mat& image, Rect image_rect) {
	Mat resized_mask, resized_mask_intersection;
	int dilate_kernel_size = 3;
	int dilate_kernel_size_2 = 61;
	double resize_ratio = 0.01;
	if (show_log)
		cout << "Start Blending ..." << endl;
	Mat dialation_kernel = getStructuringElement(MORPH_RECT, Size(dilate_kernel_size, dilate_kernel_size));
	Mat dialation_kernel_2 = getStructuringElement(MORPH_RECT, Size(dilate_kernel_size_2, dilate_kernel_size_2));

	Mat image_crop_mask = (1 - crop_stitch_image_mask);
	Mat mask_intersection;
	image_crop_mask.copyTo(mask_intersection);

	Size small_size = Size(int(image_rect.width * resize_ratio), int(image_rect.height * resize_ratio));

	resize(mask_intersection, resized_mask, small_size);
	threshold(resized_mask, resized_mask, 0.0, 1.0, THRESH_BINARY);
	resized_mask.convertTo(resized_mask, CV_8UC1);

	resize(mask_intersection, resized_mask_intersection, small_size);
	threshold(resized_mask_intersection, resized_mask_intersection, 0.0, 1.0, THRESH_BINARY);
	resized_mask_intersection.convertTo(resized_mask_intersection, CV_8UC1);
	imwrite("resized mask.jpg", resized_mask_intersection);
	int finish_blend = 0, dist = 2;
	Mat dilate_mask, inc;

	while (finish_blend == 0)
	{
		if (countNonZero(resized_mask_intersection) == resized_mask_intersection.rows * resized_mask_intersection.cols)
			finish_blend = 1;
		else
		{
			dilate(resized_mask, dilate_mask, dialation_kernel);
			bitwise_xor(dilate_mask, resized_mask, inc);
			bitwise_or(resized_mask_intersection, inc * dist, resized_mask_intersection);
			resized_mask = dilate_mask.clone();
			dist += 1;
		}
	}

	resized_mask_intersection.convertTo(resized_mask_intersection, CV_32FC1);

	dist--;
	for (int d = 2; d <= dist; d++)
	{
		for (int r = 0; r < resized_mask_intersection.rows; r++)
		{
			for (int c = 0; c < resized_mask_intersection.cols; c++)
			{
				cout << d << " " << " " << r << " " << c << endl;
				if (resized_mask_intersection.at<float>(r, c) == d)
					resized_mask_intersection.at<float>(r, c) = (float)(dist - d + 1) / dist;
			}
		}
	}

	Mat mask_intersection_new;
	resize(resized_mask_intersection, mask_intersection_new, Size(image_rect.width, image_rect.height));
	double maxVal;
	Mat dilate_mask_intersection;
	dilate(mask_intersection, dilate_mask_intersection, dialation_kernel_2);
	minMaxLoc(mask_intersection_new, NULL, &maxVal, NULL, NULL, 1 - dilate_mask_intersection);

	if (maxVal > 0)
	{
		mask_intersection_new = mask_intersection_new * (1.0 / (maxVal));
		threshold(mask_intersection_new, mask_intersection_new, 1.0, 1.0, THRESH_TRUNC);
	}

	// Map to Logistic
	double cur_value;
	for (int r = 0; r < mask_intersection_new.rows; r++)
	{
		for (int c = 0; c < mask_intersection_new.cols; c++)
		{

			cur_value = mask_intersection_new.at<float>(r, c);
			if (cur_value < 1.0)
				mask_intersection_new.at<float>(r, c) = 1.0 / (1.0 + exp(-1.5 * ((cur_value - 0.5) * 10)));
		}
	}

	Mat cur_stitch_crop =
		stitched_image.colRange(image_rect.x, image_rect.x + image_rect.width).rowRange(image_rect.y, image_rect.y + image_rect.height);

	Mat image_float, cur_stitch_crop_float;
	Mat cur_stitch_crop_layers[3], image_layers[3];
	Mat complement_weight = 1.0 - mask_intersection_new;

	cur_stitch_crop.convertTo(cur_stitch_crop_float, CV_32FC1);
	image.convertTo(image_float, CV_32FC1);

	split(cur_stitch_crop_float, cur_stitch_crop_layers);
	split(image_float, image_layers);

	for (int ii = 0; ii < 3; ii++)
	{
		cur_stitch_crop_layers[ii] = cur_stitch_crop_layers[ii].mul(complement_weight) +
			image_layers[ii].mul(mask_intersection_new);
	}
	merge(cur_stitch_crop_layers, 3, cur_stitch_crop_float);
	cur_stitch_crop_float.convertTo(cur_stitch_crop_float, CV_8UC3);
	cur_stitch_crop_float.copyTo(cur_stitch_crop);

}

Mat Stitch::stitch_and_blend(int start_row, int end_row, int start_col, int end_col, int big_tile_size, int left_margin, int top_margin, const Mat& illumination_pattern)
{
	cout << start_row << " " << start_col << endl;
	int min_r = start_tile_r[start_row][start_col], max_r = 0, min_c = start_tile_c[start_row][start_col], max_c = 0;
	for (int rr = start_row; rr <= end_row; rr++)
		for (int cc = start_col; cc <= end_col; cc++)
		{
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}

	// Exception Handling
	if (start_row < 0 || end_row < start_row || start_col < 0 || end_col < start_col)
	{
		cout << "Invalid Input for stitching...";
		return Mat();
	}

	// Calculating output width and height
	max_c += sample_width;
	max_r += sample_height;

	int output_width = max_c - min_c + 1;
	int output_height = max_r - min_r + 1;
	Mat stitched_image = Mat::zeros(Size(output_width, output_height), CV_8UC3);
	Mat cur_image, stitched_image_mask, cur_crop, cur_crop_gray;
	cout << output_width << " " << output_height << endl;
	int cur_row, cur_col;
	for (int rr = start_row; rr <= end_row; rr++)
	{
		for (int cc = start_col; cc <= end_col; cc++)
		{
			cout << "row: " << rr << " " << "col: " << cc << endl;

			cur_row = start_tile_r[rr][cc] - min_r;
			cur_col = start_tile_c[rr][cc] - min_c;

			if (load_from_disk)
				cur_image = imread(data_dir + pref + to_string(rr) + "_" + to_string(cc) + "." + image_ext, IMREAD_UNCHANGED);
			else
				cur_image = imdecode(Mat(1, lamel_images->rows[rr].columns[cc].length_, CV_8UC1, &lamel_images->rows[rr].columns[cc].data_[0]), IMREAD_UNCHANGED);
			cvtColor(cur_image, cur_image, COLOR_BGR2RGB);
			// Brightness Correction
			if (illumination_pattern.rows > 0)
			{
				Mat tiled_illu_pattern, float_image;
				Mat t[] = { illumination_pattern, illumination_pattern, illumination_pattern };
				merge(t, 3, tiled_illu_pattern);
				cur_image.convertTo(float_image, CV_32FC3);
				float_image = float_image.mul(1.0 / tiled_illu_pattern);
				float_image.convertTo(cur_image, CV_8UC3);
			}

			// Blending
			cur_crop = stitched_image.colRange(cur_col, cur_col + sample_width).rowRange(cur_row, cur_row + sample_height);
			inRange(cur_crop, Scalar(1, 1, 1), Scalar(255, 255, 255), stitched_image_mask);
			Rect cur_image_rect(cur_col, cur_row, sample_width, sample_height);
			blend(stitched_image_mask, stitched_image, cur_image, cur_image_rect);
		}
	}
	Mat bigtile_stitch_image = Mat::zeros(Size(big_tile_size, big_tile_size), CV_8UC3);

	int first_output_r = top_margin < 0 ? -top_margin : 0;
	int first_output_c = left_margin < 0 ? -left_margin : 0;

	int first_output_r_tile = top_margin < 0 ? 0 : top_margin;
	int first_output_c_tile = left_margin < 0 ? 0 : left_margin;

	int last_output_r = (output_height - first_output_r_tile + first_output_r) > big_tile_size ? big_tile_size : (output_height - first_output_r_tile + first_output_r);
	int last_output_c = (output_width - first_output_c_tile + first_output_c) > big_tile_size ? big_tile_size : (output_width - first_output_c_tile + first_output_c);

	int crop_width = last_output_c - first_output_c;
	int crop_height = last_output_r - first_output_r;


	Mat stitch_tile_crop = stitched_image.colRange(first_output_c_tile, first_output_c_tile + crop_width).
		rowRange(first_output_r_tile, first_output_r_tile + crop_height);

	stitch_tile_crop.copyTo(bigtile_stitch_image(Rect(first_output_c, first_output_r, last_output_c - first_output_c, last_output_r - first_output_r)));

	//    imshow("-1", stitched_image);
	//    imshow("2", bigtile_stitch_image);
	//    waitKey();

	return bigtile_stitch_image;
}

cv::Mat Stitch::get_illumination_pattern() {
	if (!illumination_pattern_from_hard)
	{
		int number_of_samples = 10, axis_0_valid_samples = 0, axis_1_valid_samples = 0;
		int contour_area_threshold = 2000;
		int random_index, sample_rr, sample_cc;
	    vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		Mat cur_image, image_gray, edge, foreground, image_hls, hls_component[3], image_light_mask;
		Mat axis_0_stat = Mat::zeros(1, sample_width, CV_32FC1);
		Mat axis_1_stat = Mat::zeros(sample_height, 1, CV_32FC1);

		for (int f = 0; f < number_of_samples; f++)
		{
			cout << f << endl;
			Mat foreground_reduce_0 = Mat::zeros(1, sample_width, CV_32SC1);
			Mat foreground_reduce_1 = Mat::zeros(sample_height, 1, CV_32SC1);
			Mat cur_axis_0_stat = Mat::zeros(1, sample_width, CV_32SC1);
			Mat cur_axis_1_stat = Mat::zeros(sample_height, 1, CV_32SC1);
			random_index = rand() % (row_count * column_count);
			sample_rr = random_index / column_count;
			sample_cc = random_index % column_count;
			cout << "rr: " << sample_rr << "  cc: " << sample_cc << endl;
			if (load_from_disk)
				cur_image = imread(data_dir + pref + to_string(sample_rr) + "_" + to_string(sample_cc) + "." + image_ext, IMREAD_UNCHANGED);
			else
				cur_image = imdecode(Mat(1, lamel_images->rows[sample_rr].columns[sample_cc].length_, CV_8UC1, &lamel_images->rows[sample_rr].columns[sample_cc].data_[0]), IMREAD_UNCHANGED);
			cout << "after image reading" << endl;
 			cvtColor(cur_image, cur_image, COLOR_BGR2RGB);
			medianBlur(cur_image, cur_image, 13);
			GaussianBlur(cur_image, cur_image, Size(5, 5), 21);
			cvtColor(cur_image, image_gray, COLOR_RGB2GRAY);
			Canny(image_gray, edge, 2, 7);
			dilate(edge, edge, getStructuringElement(MORPH_RECT, Size(15, 15)));
			findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
			foreground = Mat::zeros(image_gray.rows, image_gray.cols, CV_8UC1);
			for (int i = 0; i < contours.size(); i++)
			{
				if (contourArea(contours[i]) > contour_area_threshold)
					drawContours(foreground, contours, i, 255, -1, LINE_8, hierarchy);
			}
			foreground = 255 - foreground;
			cvtColor(cur_image, image_hls, COLOR_RGB2HLS);
			split(image_hls, hls_component);
			bitwise_and(foreground, hls_component[1], image_light_mask);

			reduce(image_light_mask, cur_axis_0_stat, 0, REDUCE_SUM, CV_32SC1);
			reduce(image_light_mask, cur_axis_1_stat, 1, REDUCE_SUM, CV_32SC1);
			reduce(foreground, foreground_reduce_0, 0, REDUCE_MAX, CV_8UC1);
			reduce(foreground, foreground_reduce_1, 1, REDUCE_MAX, CV_8UC1);

			if (countNonZero(foreground_reduce_0) < sample_width)
				continue;
			cur_axis_0_stat.convertTo(cur_axis_0_stat, CV_32FC1);
			for (int i = 0; i < sample_width; i++)
				axis_0_stat.at<float>(0, i) += cur_axis_0_stat.at<float>(0, i) / countNonZero(foreground.col(i));
			axis_0_valid_samples++;

			if (countNonZero(foreground_reduce_1) < sample_height)
				continue;
			cur_axis_1_stat.convertTo(cur_axis_1_stat, CV_32FC1);
			for (int i = 0; i < sample_height; i++)
				axis_1_stat.at<float>(i, 0) += cur_axis_1_stat.at<float>(i, 0) / countNonZero(foreground.row(i));
			axis_1_valid_samples++;
		}

		axis_1_stat = axis_1_stat / (float)axis_1_valid_samples;
		axis_0_stat = axis_0_stat / (float)axis_0_valid_samples;

		double maxVal;
		minMaxLoc(axis_1_stat, NULL, &maxVal, NULL, NULL, Mat());
		axis_1_stat = axis_1_stat / maxVal;
		minMaxLoc(axis_0_stat, NULL, &maxVal, NULL, NULL, Mat());
		axis_0_stat = axis_0_stat / maxVal;

		Mat illumination_pattern = axis_1_stat * axis_0_stat;
		return illumination_pattern;
	}
	else
		///TODO: Should read from a file
		return cv::Mat();
}

extern "C" __declspec(dllexport) int __cdecl get_lr_shifts(ImageRow * row_images, double* shift_r, double* shift_c)
{
	Stitch stitcher(1, row_images->column_count);
    stitcher.row_images = row_images->columns;
    auto result = stitcher.calculate_stitch_shifts_lr();
    for (int i = 0; i < row_images->column_count; i++)
	{
		shift_r[i] = result[i].shift_r;
        shift_c[i] = result[i].shift_c;
	}
	return stitcher.best_column_row.column;
}

extern "C" __declspec(dllexport) FullLamelImages __cdecl stitch_all(LamelImages* lamel_images, int* best_col, ShiftArray * shift_r, ShiftArray * shift_c)
{
	///Set Stitcher Config Arrays
	Stitch stitcher(lamel_images->row_count, lamel_images->rows->column_count);
	memcpy(&stitcher.most_informative_column[0], best_col, sizeof(int) * stitcher.row_count);

	for (int i = 0; i < stitcher.row_count; i++)
	{
		for (int j = 0; j < stitcher.column_count; j++)
		{
			stitcher.stitch_shifts_lr_row[i][j] = shift_r->rows[i].columns[j];
			stitcher.stitch_shifts_lr_col[i][j] = shift_c->rows[i].columns[j];
		}
	}

	stitcher.lamel_images = lamel_images;
	stitcher.calculate_stitch_shifts_ud();
	cout << "Vertical shifts calculated" << endl;

	/// Making Tilesets
	stitcher.Stitch_all();

	FullLamelImages full_lamel_images(stitcher.full_lamel_images);

	return full_lamel_images;
}

extern "C" __declspec(dllexport) CoordinateResult __cdecl get_big_tile_coordinates(LamelImages* lamel_images, int* best_col, ShiftArray * shift_r, ShiftArray * shift_c)
{
	///Set Stitcher Config Arrays
	Stitch stitcher(shift_r->row_count, shift_c->rows->column_count);
    memcpy(&stitcher.most_informative_column[0], best_col, sizeof(int) * stitcher.row_count);
	for (int i = 0; i < stitcher.row_count; i++)
	{
		for (int j = 0; j < stitcher.column_count; j++)
		{
			stitcher.stitch_shifts_lr_row[i][j] = shift_r->rows[i].columns[j];
			stitcher.stitch_shifts_lr_col[i][j] = shift_c->rows[i].columns[j];
		}
	}

	stitcher.lamel_images = lamel_images;
	stitcher.calculate_stitch_shifts_ud();
	cout << "Vertical shifts calculated" << endl;

	/// Making Tilesets
	vector<vector<vector<int>>> tile_config_array = stitcher.get_big_tile_coordinates();

    vector<ArrRow> arr_row;
	for (int i = 0; i < tile_config_array.size(); i++) {
		vector<IntArrCol> int_arr_cols;
		for (int j = 0; j < tile_config_array[i].size(); j++) {
			int_arr_cols.push_back(IntArrCol(tile_config_array[i][j]));
		}
		arr_row.push_back(int_arr_cols);
	}
	ThreeDArr three_d_arr(arr_row);
	vector<IntArrCol> int_arr_cols_r;
	for (int j = 0; j < stitcher.start_tile_r.size(); j++)
	{
		int_arr_cols_r.push_back(IntArrCol(stitcher.start_tile_r[j]));
	}
	ArrRow start_r(int_arr_cols_r);
    vector<IntArrCol> int_arr_cols_c;
	for (int j = 0; j < stitcher.start_tile_c.size(); j++)
	{
		int_arr_cols_c.push_back(IntArrCol(stitcher.start_tile_c[j]));
	}
	ArrRow start_c(int_arr_cols_c);
	CoordinateResult coordinate_result(three_d_arr, start_r,start_c);
	return coordinate_result;
}

extern "C" __declspec(dllexport) FullLamelImages __cdecl stitch_big_tile(LamelImages* lamel_images, CoordinateResult* coordinate_result, int idx_r, int idx_c)
{
	///Set Stitcher Config Arrays
	Stitch stitcher(lamel_images->row_count, lamel_images->rows->column_count);
	//memcpy(&stitcher.most_informative_column[0], best_col, sizeof(int) * stitcher.row_count);
	vector<vector<vector<int>>> tile_config_arr(coordinate_result->tile_config_array_.length_, vector<vector<int>>(coordinate_result->tile_config_array_.rows_->length_,vector<int>(coordinate_result->tile_config_array_.rows_->cols_->length_)));
    for (int i = 0; i < tile_config_arr.size(); i++)
	{
		for (int j = 0; j < tile_config_arr[i].size(); j++)
		{
			for (int k = 0; k < tile_config_arr[i][j].size(); k++)
			{
				tile_config_arr[i][j][k] = coordinate_result->tile_config_array_.rows_[i].cols_[j].elements_[k];
			}
		}
	}
	stitcher.tile_config_array = tile_config_arr;
	stitcher.lamel_images = lamel_images;
	for (int i = 0; i <lamel_images->row_count; i++)
	{
		for (int j = 0; j <lamel_images->rows->column_count; j++)
		{
			stitcher.start_tile_r[i][j] = coordinate_result->start_r_.cols_[i].elements_[j];
			stitcher.start_tile_c[i][j] = coordinate_result->start_c_.cols_[i].elements_[j];
		}
	}
    /// Making Tilesets
	stitcher.Stitch_big_tile(idx_r, idx_c);

	FullLamelImages full_lamel_images(stitcher.full_lamel_images);

	return full_lamel_images;
}
