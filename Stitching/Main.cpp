#include "Stitch.h"

int main()
{
	int row_count = 60;
	int column_count = 87;

	Stitch s(row_count,column_count);
	
    for (int i = 0; i < row_count; i++)
		/// When a images of row i is completely received
	  s.stitch_shifts_lr[i] = s.calculate_stitch_shifts_lr(i);

	s.calculate_stitch_shifts_ud();
	s.Stitch_all();
}