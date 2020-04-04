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
    ActivationFunctionType getActivationFunction(std::string value);
    int getInputsOutputs(std::string value);
    std::vector<std::string> getProperties(std::string value);
    void printConfig();
    std::vector<std::string> splitString(const std::string &s, char delimiter);

    std::vector<LayerType> layers;
    std::vector<std::map<std::string, std::string>> layerProperties;
    std::map<std::string, std::string> propertiesMap;

private:
};

#endif