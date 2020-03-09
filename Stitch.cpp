#include "Stitch.h"
#include <cliext/vector>

using namespace std;
namespace CLI
{
Stitch::Stitch(int row_count, int column_count)
    :ManagedObject(new core::Stitch(row_count,column_count))
{}

vector<vector<core::Stitch::shift>> Stitch::calculate_stitch_shifts_lr_in_column()
{
	return m_Instance->calculate_stitch_shifts_lr_in_column();
}

vector<core::Stitch::shift> Stitch::calculate_stitch_shifts_ud()
{
	return m_Instance->calculate_stitch_shifts_ud();
}

}