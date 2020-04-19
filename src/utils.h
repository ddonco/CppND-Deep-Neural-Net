#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <regex>

#include "activation.h"
#include "layer.h"

class Config
{
public:
    void readConfig(std::string configPath);
    LayerType getLayer(std::string key);
    void printConfig();

    std::vector<LayerType> layers;
    std::vector<std::map<std::string, std::string>> layerProperties;
    std::map<std::string, std::string> propertiesMap;

private:
};

namespace Utils
{
ActivationFunctionType parseActivationFunction(std::string value);
int parseInputsOutputs(std::string value);
std::vector<std::string> parseProperties(std::string value);
std::vector<std::string> splitString(const std::string &s, char delimiter);
}; // namespace Utils

template <typename M>
M readCsv(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime>>(values.data(), rows, values.size() / rows);
}

#endif