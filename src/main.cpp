#include "utils.h"

int main()
{
    std::string configPath = "../config/l3.config";

    Config networkConfig;
    networkConfig.readConfig(configPath);
    networkConfig.printConfig();

    return 0;
}