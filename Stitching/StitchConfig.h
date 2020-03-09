#include <boost/optional.hpp>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace pt = boost::property_tree;

std::string CONFIG_FILE_PATH = "StitchConfig.json";


class StringConfig
{
public:
	StringConfig();
	pt::ptree root;
	
};

