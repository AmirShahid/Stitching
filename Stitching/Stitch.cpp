#include "Stitch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> 
#include <conio.h>
#include <time.h>
#include <string> 
#include <cmath> 
#include <boost/filesystem.hpp>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <future>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;
namespace pt = boost::property_tree;

Stitch::Stitch(int row_count, int column_count, int zoom_levels) :
	stitch_shifts_lr(row_count, vector<shift>(column_count, shift())),
	stitch_shifts_lr_row(row_count, vector<double>(column_count, double())),
	stitch_shifts_lr_col(row_count, vector<double>(column_count, double())),
	//stitched_images(zoom_levels, vector<vector<ImageFile>>(row_count, vector<ImageFile>(column_count, ImageFile()))),
	stitch_shifts_ud(row_count, shift()),
	stitch_shifts_ud_row(row_count),
	stitch_shifts_ud_col(row_count),
	stitch_shifts_row(column_count),
	best_column_for_ud(row_count, best_column()),
	most_informative_column(row_count),
	start_tile_r(row_count, vector<int>(column_count, 0)),
	start_tile_c(row_count, vector<int>(column_count, 0)),
    zoom_levels(zoom_levels-1),
    row_count(row_count),
    column_count(column_count)
{
	cout << "constructor called" << endl;
	for (int i = -1 * vertical_deviation; i <= vertical_deviation; i++) shift_idx.push_back(i);
	pt::read_json(CONFIG_FILE_PATH, config);
    vertical_deviation = config.get<int>("vertical_deviation");
    number_of_threads = config.get<int>("number_of_threads");
    data_dir = config.get<string>("data_dir");
	dataset_name = config.get<string>("dataset_name");
	path_delim = config.get<string>("path_delim");
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
	instance_count++;
}

float round_n(float num, int dec)
{
	double m = (num < 0.0) ? -1.0 : 1.0;   // check if input is negative
	double pwr = pow(10, dec);
	return float(floor((double)num * m * pwr + 0.5) / pwr) * m;
}

void shift_cleaner(vector<Stitch::shift>& shifts) {
	vector<double> row;
	vector<double> col;
	for (auto value : shifts)  if (value.shift_r != INVALID_VALUE && value.shift_c != INVALID_VALUE) 
	{
	    row.push_back(value.shift_r);
		col.push_back(value.shift_c);
	}

    std::nth_element(row.begin(), row.begin() + row.size() / 2, row.end());
    std::nth_element(col.begin(), col.begin() + col.size() / 2, col.end());
 /*   std::cout << "The median is " << v[v.size() / 2] << '\n';*/


    for (auto& shift : shifts)
    {
	    if (shift.shift_r == INVALID_VALUE )
		    shift.shift_r = row[row.size() / 2];
		if (shift.shift_c == INVALID_VALUE)
			shift.shift_c = col[col.size() / 2];
    }
}
double calc_median(vector<double> shifts)
{
	vector<double> new_shifts;
	size_t size = shifts.size();
	for (double v : shifts) if (v != INVALID_VALUE) new_shifts.push_back(v);
	
    sort(new_shifts.begin(), new_shifts.end());
	size = new_shifts.size();
    if (size == 0)
    {
		return INVALID_VALUE;
    }
	if (size % 2 == 0)
	{
		//cout << (new_shifts[size / 2 - 1] + new_shifts[size / 2]) / 2 << endl;
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

	threshold(absdiff_bg, absdiff_bg, 2 * grey_threshold, 255.0, CV_THRESH_BINARY);
	threshold(absdiff_gr, absdiff_gr, 2 * grey_threshold, 255.0, CV_THRESH_BINARY);
	threshold(absdiff_br, absdiff_br, 2 * grey_threshold, 255.0, CV_THRESH_BINARY);

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
	//std::cout << "COLORNESS_RATIO: " << color_ratio << endl;
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
	
	cvtColor(image_up, image_up_gray, CV_BGR2GRAY);
	cvtColor(image_down, image_down_gray, CV_BGR2GRAY);
	int effective_y = (int)(image_up_gray.rows * lu_image_portion);
	image_up_gray(Rect(0, effective_y, image_up_gray.cols, (int)(image_up_gray.rows - effective_y))).copyTo(image_up_gray_cropped);

    vector<double> shift_direction_r;
	vector<double> shift_direction_c;
	
	int step_r = (int)(image_down.rows / split_ratio_ud);
	int step_c = (int)(image_down.cols / split_ratio_ud);

	// Check Blankness

	blank_property blank_stat = is_image_blank(image_down(Rect(0, 0, image_down.cols, step_r)),grey_std_thresh_ST);
	if (blank_stat.is_blank)
	{
		//std::cout << stitch_shift.shift_r << "  " << stitch_shift.shift_c << "BLANK" << endl;
		return shift(INVALID_VALUE,INVALID_VALUE);
	}
	else
	{
		//image_up_gray_cropped.convertTo(image_up_gray_cropped, CV_32FC1);
		cv::Range range[2];
		for (int j = 0; j < split_ratio_ud; j++)
		{
			if (shift_direction_c.size() >= maxHighCorrPoint_ud) break;

			range[0] = Range(0, (long)step_r);
			range[1] = Range((long)(step_c * j), (long)(step_c * (j + 1)));
			image_cur_split = image_down_gray(range);
			//image_cur_split.convertTo(image_cur_split, CV_32FC1);

			matchTemplate(image_up_gray_cropped, image_cur_split, corrMatrix, TM_CCOEFF_NORMED);
			minMaxLoc(corrMatrix, &min_value_corr, &max_value_corr, &min_loc_corr, &max_loc_corr, Mat());
			//std::cout << "Max Corr Value: " << round_n(max_value_corr, 2) << endl;
			if(max_value_corr > acceptanceThresh_ud)
			{
				match_loc = max_loc_corr;
				match_loc.y += effective_y;
				center_c = (long)(image_down.cols / split_ratio_ud * j);
				if (((float) abs((match_loc.x - center_c) - expected_shift_c_ud)) > max_shift_c_threshold_ud ||
					((float) abs((match_loc.y - expected_shift_r_ud))) > max_shift_r_threshold_ud) continue;

				shift_direction_r.push_back((double)(match_loc.y));
				shift_direction_c.push_back((double)(match_loc.x - center_c));

			}
		}
		if (shift_direction_c.empty())
		{
			//std::cout << stitch_shift.shift_r << "  " << stitch_shift.shift_c << "NO DETECTTION" << endl;
			return shift(INVALID_VALUE,INVALID_VALUE);
		}
		else
		{
			stitch_shift.shift_c = (double)(calc_median(shift_direction_c));
			stitch_shift.shift_r = (double)(calc_median(shift_direction_r));
        
			//cout << "Row:  " << stitch_shift.shift_r << " Col:  " << stitch_shift.shift_c << endl;
			return stitch_shift;
		}
	}
}


Stitch::shift Stitch::stitch_lr(Mat& image_left, Mat& image_right) {

	clock_t ttt = clock();

	if (image_left.empty() || image_right.empty())
		return shift(INVALID_VALUE,INVALID_VALUE);

	shift stitch_shift;
	double max_value_corr, min_value_corr;
	int center_r;
	Point min_loc_corr, max_loc_corr, match_loc;
	Mat image_left_gray, image_left_gray_cropped, image_right_gray, image_cur_split, corrMatrix;
	cvtColor(image_left, image_left_gray, CV_BGR2GRAY);
	cvtColor(image_right, image_right_gray, CV_BGR2GRAY);
	int effective_x = (int)(image_left_gray.cols * lu_image_portion);
	image_left_gray(Rect(effective_x, 0, (int)(image_left_gray.cols - effective_x), image_left_gray.rows)).copyTo(image_left_gray_cropped);

	// init list and parameters
	vector<double> shift_direction_r;
	vector<double> shift_direction_c;
	int temp_r = (int)(image_right_gray.rows / split_ratio_lr);
	int temp_c = (int)(image_right_gray.cols / split_ratio_lr);
    
	// Check Blankness
	//cout << "block_stitch_init_time: " << clock() - ttt << endl;
	//ttt = clock();

	blank_property blank_stat = is_image_blank(image_right(Rect(0, 0, temp_c, image_right.rows)),grey_std_thresh_ST);
	
    if (blank_stat.is_blank)
	{
		//std::cout << stitch_shift.shift_r << "  " << stitch_shift.shift_c << "BLANK" << endl;
		//cout << "block_stitch_isBlank_time: " << clock() - ttt << endl;
		return shift();
	}
	else
	{
		cv::Range range[2];
		//cout << "block_stitch_preStitch_time: " << clock() - ttt << endl;

		for (int i = 0; i < split_ratio_lr && shift_direction_c.size() < maxHighCorrPoint_lr; i++)
		{
			//ttt = clock();

			range[0] = Range((long)(temp_r * i), (long)(temp_r * (i + 1)));
			range[1] = Range(0, (long)(temp_c));
			image_cur_split = image_right_gray(range);
			//imshow("tmpl" + to_string(i), image_cur_split);
			//waitKey(0);

			matchTemplate(image_left_gray_cropped, image_cur_split, corrMatrix, TM_CCOEFF_NORMED);
			minMaxLoc(corrMatrix, &min_value_corr, &max_value_corr, &min_loc_corr, &max_loc_corr, Mat());
			//cout << "block_stitch_matchTemp_time: " << clock() - ttt << endl;
			if (max_value_corr > acceptanceThresh_lr)
			{
				//ttt = clock();
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
			//std::cout << stitch_shift.shift_r << "   " << stitch_shift.shift_c << "NO DETECTTION" << endl;
			return shift();
		}
		else
		{
			ttt = clock();
			stitch_shift.shift_c = (double)(calc_median(shift_direction_c));
			stitch_shift.shift_r = (double)(calc_median(shift_direction_r));

			//cout << "Row:  " <<  stitch_shift.shift_r << " Col:  " <<  stitch_shift.shift_c << endl;
			//cout << "block_stitch_calcMedian_time: " << clock() - ttt << endl;
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
		//image_up = imread(data_dir + path_delim + dataset_name + path_delim + pref + to_string(i) + "_" + to_string(best_column_for_ud[i].column) + "." + image_ext, IMREAD_UNCHANGED);
		cout << "received image size " << lamel_images->rows[i].columns[most_informative_column[i]].length << endl;
		image_up = imdecode(Mat(1, lamel_images->rows[i].columns[most_informative_column[i]].length, CV_8UC1, lamel_images->rows[i].columns[most_informative_column[i]].data), CV_LOAD_IMAGE_UNCHANGED);
	    for (auto H_shift : shift_idx)
		{
			//image_down = imread(data_dir + path_delim + dataset_name + path_delim + pref + to_string(i + 1) + "_" +
				//to_string(best_column_for_ud[i].column + H_shift) + "." + image_ext, IMREAD_UNCHANGED);
			image_down = imdecode(Mat(1, lamel_images->rows[i+1].columns[most_informative_column[i]].length, CV_8UC1, lamel_images->rows[i + 1].columns[most_informative_column[i]].data), CV_LOAD_IMAGE_UNCHANGED);
	        shift stitch_result = stitch_ud(image_up, image_down);

			temp_shift_r[H_shift + vertical_deviation] = stitch_result.shift_r;
			temp_shift_c[H_shift + vertical_deviation] = stitch_result.shift_c;

			if (temp_shift_c[H_shift + vertical_deviation] != INVALID_VALUE)
			{
				if (H_shift < 0)
				{
                    for (int k = H_shift + most_informative_column[i]; k < most_informative_column[i]; k++)
                    {
                        if (stitch_shifts_lr_row[i][k] != INVALID_VALUE)
						    temp_shift_r[H_shift + vertical_deviation] += stitch_shifts_lr_row[i][k];
                        if (stitch_shifts_lr_col[i][k] != INVALID_VALUE)
						    temp_shift_c[H_shift + vertical_deviation] += stitch_shifts_lr_col[i][k];
                    }
				}
				else if (H_shift > 0)
				{
					for (int k = H_shift + most_informative_column[i]; k > most_informative_column[i]; k--)
					{
						if (stitch_shifts_lr_row[i][k] != INVALID_VALUE)
							temp_shift_r[H_shift + vertical_deviation] -= stitch_shifts_lr_row[i][k];
						if (stitch_shifts_lr_col[i][k] != INVALID_VALUE)
							temp_shift_c[H_shift + vertical_deviation] -= stitch_shifts_lr_col[i][k];
					}
				}
			}
		}
        const double median_r = calc_median(temp_shift_r);
        const double median_c = calc_median(temp_shift_c);
		stitch_shifts_ud[i] = shift(median_r, median_c);
	    stitch_shifts_ud_row[i] = median_r;
	    stitch_shifts_ud_col[i] = median_c;
	}
}

void Stitch::stitch_single_thread_lr(int start_row, int end_row, int start_col, int end_col)
{
	Mat image_left, image_right;
	blank_property blank_stat;
    for (auto i = start_row; i <= end_row ;i++)
    {
    for (auto j = start_col; j < end_col; j++)
	{
		if (!image_right.empty())
			image_left = image_right;
		else
			image_left = imdecode(Mat(1, row_images[j].length, CV_8UC1, row_images[j].data), CV_LOAD_IMAGE_UNCHANGED);
            //image_left = imread(data_dir + path_delim + dataset_name + path_delim + pref + to_string(i) + "_" + to_string(j) + "." + image_ext);
		//image_right = imread(data_dir + path_delim + dataset_name + path_delim + pref + to_string(i) + "_" + to_string(j + 1) + "." + image_ext);
		image_right = imdecode(Mat(1, row_images[j+1].length, CV_8UC1, row_images[j+1].data), CV_LOAD_IMAGE_UNCHANGED);
        //stitch_shifts_lr[i][j] = stitch_lr(image_left, image_right);
        stitch_shifts_row[j] = stitch_lr(image_left, image_right);
        blank_stat = is_image_blank(image_left(Rect(0, (int)(lu_image_portion * image_left.rows), image_left.cols, image_left.rows - (int)(lu_image_portion * image_left.rows))),grey_std_thresh_BC);
		//if (best_column_for_ud[start_row].color_ratio < blank_stat.color_ratio)
			//best_column_for_ud[start_row] = best_column(j, blank_stat.color_ratio);
		if (best_column_row.color_ratio < blank_stat.color_ratio)
			best_column_row = best_column(j, blank_stat.color_ratio);
		cout << instance_count << " , " << j << " color_ratio: " << blank_stat.color_ratio << endl;
	}
    }
}

//vector<vector<Stitch::shift>> Stitch::calculate_stitch_shifts_lr_in_row()
//{
//	array<shift,5> a;
//
//	vector<future<void>> lr_threads;
//    for (int i = 0; i < row_count; i = i + number_of_threads)
//    {
//		for (int j = 0; j < number_of_threads; j++)
//		{
//            lr_threads.push_back(async(launch::async,&Stitch::stitch_single_thread_lr,this,i+j, i+j, 0, column_count-1));
//		}
//		for (auto &thread : lr_threads)
//		{
//			thread.wait();
//		}
//    }
//	return stitch_shifts_lr;
//}
//
//vector<vector<Stitch::shift>> Stitch::calculate_stitch_shifts_lr_in_column()
//{
//    vector<future<void>> lr_threads;
//	for (int i = 0; i < row_count; i++)
//	{
//		int offset = column_count/number_of_threads;
//        for (int j = 0; j < column_count; j = j + offset)
//		{
//			lr_threads.push_back(async(launch::async, &Stitch::stitch_single_thread_lr ,this,i, i, j, min((j + offset + 1), column_count-1)));
//		}
//	    for (auto &thread : lr_threads)
//	    {
//		    thread.wait();
//	    }
//		lr_threads.clear();
//	}
//	return stitch_shifts_lr;
//}

vector<Stitch::shift>& Stitch::calculate_stitch_shifts_lr(int row_number)
{
    vector<future<void>> lr_threads;
	int offset = column_count/number_of_threads;
    for (int j = 0; j < column_count; j = j + offset)
	{
		lr_threads.push_back(async(launch::async, &Stitch::stitch_single_thread_lr ,this, row_number, row_number, j, min((j + offset + 1), column_count-1)));
	}
	for (auto &thread : lr_threads)
	{
		thread.wait();
	}
 	lr_threads.clear();
	//shift_cleaner(stitch_shifts_lr[row_number]);
	shift_cleaner(stitch_shifts_row);
	/*pt::ptree json_arr;
	for (int i = 0; i < stitch_shifts_row.size(); i++)
	{
		pt::ptree shift_pair;
		shift_pair.put("shift_r", stitch_shifts_row[row_number][i].shift_r);
		shift_pair.put("shift_c", stitch_shifts_row[row_number][i].shift_c);
		json_arr.push_back(make_pair(to_string(i), shift_pair));
	}*/
	/*pt::ptree root;
	root.add_child(to_string(row_number),json_arr);
	pt::ptree best_col;
	best_col.put("column", best_column_for_ud[row_number].column);
	best_col.put("color_ratio", best_column_for_ud[row_number].color_ratio);
	root.add_child("most_informative_column",best_col);
	pt::write_json("LR_" + to_string(row_number) + ".json", root);*/
	return stitch_shifts_row;
}

vector<Stitch::shift>& Stitch::calculate_stitch_shifts_ud()
{
	pt::ptree lrS;
    for (int i = 0; i < row_count; i++)
    {

		 cout << "lr_ "+to_string(i)+" shifts read from json" << endl;
		 pt::read_json("LR_" + to_string(i) + ".json",lrS);
		 for(int j = 0 ; j < column_count ; j++)
         {
			 stitch_shifts_lr_row[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_r");
			 stitch_shifts_lr_col[i][j] = lrS.get<double>(to_string(i) + "." + to_string(j) + "." + "shift_c");
			 stitch_shifts_lr[i][j] = shift(lrS.get<double>(to_string(i)+"."+to_string(j) + "." + "shift_r"), lrS.get<double>(to_string(i)+"."+to_string(j) + "." + "shift_c"));
         }
		 most_informative_column[i] = lrS.get<int>("most_informative_column.column");
		 best_column_for_ud[i] = best_column(lrS.get<int>("most_informative_column.column") ,lrS.get<double>("most_informative_column.color_ratio"));
    }
	vector<future<void>> ud_threads;
	int offset = row_count / number_of_threads;
	for (int j = 0; j < row_count; j = j + offset)
	{
		ud_threads.push_back(async(launch::async, &Stitch::stitch_single_thread_ud, this, j, min(j + offset + 1, row_count - 1)));
	}
	for (auto &thread : ud_threads)
	{
		thread.wait();
	}
	ud_threads.clear();
	shift_cleaner(stitch_shifts_ud);
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
	return stitch_shifts_ud;
}

void update_min_and_max(int new_value, int &min_val, int &max_val) {
	if (new_value > max_val)
		max_val = new_value;
	if (new_value < min_val)
		min_val = new_value;
}



void Stitch::Stitch_all()
{
	pt::ptree lrS;
	pt::ptree udS;
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
		stitch_shifts_ud_row[i] = udS.get<double>(to_string(i) + ".shift_r");
		stitch_shifts_ud_col[i] = udS.get<double>(to_string(i) + ".shift_c");
		best_column_for_ud[i] = best_column(lrS.get<int>("most_informative_column.column"), lrS.get<double>("most_informative_column.color_ratio"));
		most_informative_column[i] = lrS.get<int>("most_informative_column.column");
	}
    
	int start_row = 0, end_row = 54, start_col = 5, end_col = 80;
		int min_r = 0, max_r = 0, min_c = 0, max_c = 0;
	enum tile_config { START_ROW, END_ROW, START_COL, END_COL, LEFT_MARGIN, TOP_MARGIN };

	for (int cc = start_col + 1; cc <= end_col; cc++)
	{
		start_tile_r[start_row][cc] = start_tile_r[start_row][cc - 1] + stitch_shifts_lr_row[start_row][cc - 1];
		start_tile_c[start_row][cc] = start_tile_c[start_row][cc - 1] + stitch_shifts_lr_col[start_row][cc - 1];
		update_min_and_max(start_tile_r[start_row][cc], min_r, max_r);
		update_min_and_max(start_tile_c[start_row][cc], min_c, max_c);
	}

	for (int rr = start_row + 1; rr <= end_row; rr++)
	{
		start_tile_r[rr][best_column_for_ud[rr - 1].column] =
			start_tile_r[rr - 1][best_column_for_ud[rr - 1].column] + stitch_shifts_ud_row[rr - 1];

		start_tile_c[rr][best_column_for_ud[rr - 1].column] =
			start_tile_c[rr - 1][best_column_for_ud[rr - 1].column] + stitch_shifts_ud_col[rr - 1];

		update_min_and_max(start_tile_r[rr][best_column_for_ud[rr - 1].column], min_r, max_r);
		update_min_and_max(start_tile_c[rr][best_column_for_ud[rr - 1].column], min_c, max_c);

		for (int cc = best_column_for_ud[rr - 1].column + 1; cc <= end_col; cc++)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][best_column_for_ud[rr - 1].column];
			start_tile_c[rr][cc] = start_tile_c[rr][best_column_for_ud[rr - 1].column];
			for (int k = best_column_for_ud[rr - 1].column; k<cc; k++)
			{
				start_tile_r[rr][cc] += stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] += stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}

		for (int cc = best_column_for_ud[rr - 1].column - 1; cc >= start_col; cc--)
		{
			start_tile_r[rr][cc] = start_tile_r[rr][best_column_for_ud[rr - 1].column];
			start_tile_c[rr][cc] = start_tile_c[rr][best_column_for_ud[rr - 1].column];
			for (int k = best_column_for_ud[rr - 1].column - 1; k >= cc; k--)
			{
				start_tile_r[rr][cc] -= stitch_shifts_lr_row[rr][k];
				start_tile_c[rr][cc] -= stitch_shifts_lr_col[rr][k];
			}
			update_min_and_max(start_tile_r[rr][cc], min_r, max_r);
			update_min_and_max(start_tile_c[rr][cc], min_c, max_c);
		}
	}
	// Calculating output width and height
	
	int big_tile_size = int(tile_size * pow(2, zoom_levels));
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
	cout << "output width: " << output_width << " " << "output Height: " << output_height << endl;
	int number_of_big_tile_c = output_width / big_tile_size + 1;
	int number_of_big_tile_r = output_height / big_tile_size + 1;

	vector<vector<vector<int>>> tile_config_array(number_of_big_tile_r,vector<vector<int>>(number_of_big_tile_c, vector<int>(6,INVALID_VALUE)));

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
					tile_config_array[idx_r][idx_c][TOP_MARGIN]);
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
			    for (int dd = 0; dd<(zoom_levels - zz); dd++)
				{
					resize(big_tile_clone, big_tile_clone, Size(int(big_tile_clone.cols / 2), int(big_tile_clone.rows / 2)));
				}
				for (int yy = 0; yy<int(big_tile_clone.rows / tile_size); yy++)
				{
					for (int xx = 0; xx<int(big_tile_clone.cols / tile_size); xx++)
					{
						current_tile = big_tile_clone.colRange(xx*tile_size, (xx + 1) * tile_size)
							.rowRange(yy*tile_size, (yy + 1) * tile_size);
						/*imshow("2", current_tile);
						waitKey();*/
						// Z = zz | Y = row_bias + yy | X = col_bias + xx
						vector<uchar> encoded_image;
						imencode(".jpeg", current_tile, encoded_image);
					    ImageFile image_file(&encoded_image[0],encoded_image.size());
						full_lamel_images.push_back(FullLamelImage(&image_file, col_bias + xx, row_bias + yy,zz));
					}
				}
			}
		}
	}
	cout << "END...." << endl;
}

void Stitch::blend(Mat& crop_stitch_image_mask, Mat& stitched_image, Mat& image, Rect image_rect) {
	Mat resized_mask, resized_mask_intersection;
	int dilate_kernel_size = 3;
	int dilate_kernel_size_2 = 61;
	double resize_ratio = 0.01;
	//cout << "Start Blending ..." << endl;
	Mat dialation_kernel = getStructuringElement(MORPH_RECT, Size(dilate_kernel_size, dilate_kernel_size));
	Mat dialation_kernel_2 = getStructuringElement(MORPH_RECT, Size(dilate_kernel_size_2, dilate_kernel_size_2));

	Mat image_crop_mask = (1 - crop_stitch_image_mask);
	Mat mask_intersection;
	image_crop_mask.copyTo(mask_intersection);

	Size small_size = Size(int(image_rect.width * resize_ratio), int(image_rect.height * resize_ratio));

	resize(mask_intersection, resized_mask, small_size);
	threshold(resized_mask, resized_mask, 0.0, 1.0, CV_THRESH_BINARY);
	resized_mask.convertTo(resized_mask, CV_8UC1);

	resize(mask_intersection, resized_mask_intersection, small_size);
	threshold(resized_mask_intersection, resized_mask_intersection, 0.0, 1.0, CV_THRESH_BINARY);
	resized_mask_intersection.convertTo(resized_mask_intersection, CV_8UC1);

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
			bitwise_or(resized_mask_intersection, inc*dist, resized_mask_intersection);
			resized_mask = dilate_mask.clone();
			dist += 1;
		}
	}

	resized_mask_intersection.convertTo(resized_mask_intersection, CV_32FC1);

	dist--;
	for (int d = 2; d <= dist; d++)
		for (int r = 0; r < resized_mask_intersection.rows; r++)
			for (int c = 0; c < resized_mask_intersection.cols; c++)
			{
				if (resized_mask_intersection.at<float>(r, c) == d)
					resized_mask_intersection.at<float>(r, c) = (float)(dist - d + 1) / dist;
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

	for (int ii = 0; ii<3; ii++)
		cur_stitch_crop_layers[ii] = cur_stitch_crop_layers[ii].mul(complement_weight) +
		image_layers[ii].mul(mask_intersection_new);
	merge(cur_stitch_crop_layers, 3, cur_stitch_crop_float);
	cur_stitch_crop_float.convertTo(cur_stitch_crop_float, CV_8UC3);
	cur_stitch_crop_float.copyTo(cur_stitch_crop);

}


Mat Stitch::stitch_and_blend(int start_row, int end_row, int start_col, int end_col, int big_tile_size, int left_margin, int top_margin)
{
	cout << "start" << endl;
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

	if (min_c + left_margin + big_tile_size > max_c)
		max_c = min_c + left_margin + big_tile_size;
	if (min_r + top_margin + big_tile_size > max_r)
		max_r = min_r + top_margin + big_tile_size;
	if (left_margin < 0)
	{
		min_c += left_margin;
		left_margin = 0;
	}
	if (top_margin < 0)
	{
		min_r += top_margin;
		top_margin = 0;
	}

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

			cur_image = imread(data_dir + path_delim + dataset_name + path_delim + pref + to_string(rr) + "_" + to_string(cc) + "." + image_ext, CV_LOAD_IMAGE_UNCHANGED);
			//cur_image = imdecode(Mat(1, lamel_images->rows[rr].columns[cc].length, CV_8UC1, lamel_images->rows[rr].columns[cc].data), CV_LOAD_IMAGE_UNCHANGED);
			cvtColor(cur_image, cur_image, COLOR_BGR2RGB);
			//cout << "image "+ to_string(rr) + " " + to_string(cc) +" read" << endl;
		    // Blending
			cur_crop = stitched_image.colRange(cur_col, cur_col + sample_width).rowRange(cur_row, cur_row + sample_height);
			inRange(cur_crop, Scalar(1, 1, 1), Scalar(255, 255, 255), stitched_image_mask);
			Rect cur_image_rect(cur_col, cur_row, sample_width, sample_height);
			blend(stitched_image_mask, stitched_image, cur_image, cur_image_rect);
		}
	}

	return stitched_image.colRange(left_margin, left_margin + big_tile_size)
		.rowRange(top_margin, top_margin + big_tile_size);
}



extern "C" __declspec(dllexport) int __cdecl get_lr_shifts(ImageRow* row_images, double* shift_r, double* shift_c)
{
    Stitch stitcher(0,row_images->column_count);
	stitcher.row_images = row_images->columns;
	auto result = stitcher.calculate_stitch_shifts_lr();
    for (int i = 0; i < row_images->column_count; i++)
    {
		shift_r[i] = result[i].shift_r;
		shift_c[i] = result[i].shift_c;
    }
	return stitcher.best_column_row.column;
}

extern "C" __declspec(dllexport) FullLamelImages __cdecl stitch_all(LamelImages* lamel_images, int* best_col, ShiftArray* shift_r, ShiftArray* shift_c)
{
    ///Set Stitcher Arrays
    Stitch stitcher(lamel_images->row_count,lamel_images->rows->column_count,6);
	//memcpy(&stitcher.most_informative_column[0],best_col,sizeof(int)*stitcher.row_count);
 //   for (int i = 0; i < stitcher.row_count; i++)
 //   {
	//	for (int j = 0; j < stitcher.column_count; j++)
	//	{
	//		/*stitcher.stitch_shifts_lr[i][j] = Stitch::shift(shift_r->rows[i].columns[j], shift_c->rows[i].columns[j]);*/
	//		stitcher.stitch_shifts_lr_row[i][j] = shift_r->rows[i].columns[j];
	//		stitcher.stitch_shifts_lr_col[i][j] = shift_c->rows[i].columns[j];
	//	}
 //   }
	stitcher.lamel_images = lamel_images;
    //stitcher.calculate_stitch_shifts_ud();
	//cout << "ud ended" << endl;
    /// Making Tilesets
	stitcher.Stitch_all();
	cout << "length [5 0 0]" << stitcher.full_lamel_images[0].x << endl;
	//ImageRow stitched_image_row;
	//stitched_image_row.columns = &stitcher.stitched_images[0][0][0];
	//stitched_image_row.column_count = stitcher.stitched_images[0][0].size();
	//LamelImages stitched_lamel_images;
	//stitched_lamel_images.rows = &stitched_image_row;
	//stitched_lamel_images.row_count = stitcher.stitched_images[0].size();
    /*FullLamelLevels full_lamel_levels;
	full_lamel_levels.lamel_images = &stitched_lamel_images;
	full_lamel_levels.zoom_level_count = stitcher.stitched_images.size();*/
	FullLamelImages full_lamel_images;
	full_lamel_images.full_lamel_image = &stitcher.full_lamel_images[0];
	full_lamel_images.length = stitcher.full_lamel_images.size();
	cout << full_lamel_images.length << endl;
	return full_lamel_images;
}