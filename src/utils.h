#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <regex>

#include "layer.h"

class Config
{
public:
    void readConfig(std::string configPath);
    LayerType getLayer(std::string key);
    ActivationFunctionType getActivationFunction(std::string key);
    void printConfig();

    std::vector<LayerType> layers;
    std::vector<std::vector<std::string>> layerProperties;

private:
};

#endif