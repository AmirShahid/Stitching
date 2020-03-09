#include "Stitch.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

int main()
{
	double shift_r[60];
	double shift_c[60];
	Stitch s(60,87,6);
    //for (int i=0;i<60;i++)
	   // s.calculate_stitch_shifts_lr(i);
    //s.calculate_stitch_shifts_ud();
	s.Stitch_all();
}