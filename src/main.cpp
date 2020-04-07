#include "utils.h"
#include "model.h"

int main()
{
    std::string configPath = "../config/l3.config";

    Config networkConfig;
    networkConfig.readConfig(configPath);
    // networkConfig.printConfig();

    Model model = Model(networkConfig);
    model.printModel();

    model.testForwardPass();

    return 0;
}