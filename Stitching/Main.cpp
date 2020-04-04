#include "Stitch.h"
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    /// This code is just for testing cpp before making the dll 
    
	int row_count = 40;
	int column_count = 40;
	
	Stitch s(row_count,column_count);
	
    for (int i = 0; i < row_count; i++)
		 //When row ith images of lamel captured completely
	  s.calculate_stitch_shifts_lr(i);

	s.calculate_stitch_shifts_ud();
	s.Stitch_all();

    
    for (auto full_lamel_image : s.full_lamel_images)
    {
		string path = "./" + to_string(full_lamel_image.z_) + "/" + std::to_string(full_lamel_image.x_) + "_" + std::to_string(full_lamel_image.y_) + ".jpg";
		imwrite(path, imdecode(Mat(1, full_lamel_image.image_file_.length_, CV_8UC1, full_lamel_image.image_file_.data_), CV_LOAD_IMAGE_UNCHANGED));
    }
    
}
