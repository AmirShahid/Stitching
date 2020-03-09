#include "StitchConfig.h"


StringConfig::StringConfig()
{
	pt::read_json("file.json", root);
}
