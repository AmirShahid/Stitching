#include "Stitch.h"

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    /// This code is just for testing cpp before making the dll 
    
	int row_count = 20;
	int column_count = 10;
	
	Stitch s(row_count,column_count);
	
    //for (int i = 0; i < row_count; i++)
	   // //When row ith images of lamel captured completely
	   // s.calculate_stitch_shifts_lr(i);

	//s.calculate_stitch_shifts_ud();
	//s.Stitch_all();
	vector<vector<vector<int>>> tile_config_array = s.get_big_tile_coordinates();
	s.tile_config_array = tile_config_array;
	s.Stitch_big_tile(0, 0);

    for (auto full_lamel_image : s.full_lamel_images)
    {
		string path = "./" + to_string(full_lamel_image.z_) + "_" + std::to_string(full_lamel_image.x_) + "_" + std::to_string(full_lamel_image.y_) + ".jpg";
		imwrite(path, imdecode(Mat(1, full_lamel_image.image_file_.length_, CV_8UC1, full_lamel_image.image_file_.data_), IMREAD_UNCHANGED));
    }
    
}
