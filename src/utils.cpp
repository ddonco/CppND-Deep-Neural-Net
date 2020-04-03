#include "utils.h"

void Config::readConfig(std::string configPath)
{
    std::string line, value;
    std::vector<std::string> properties;
    std::ifstream filestream(configPath);
    if (filestream.is_open())
    {
        while (std::getline(filestream, line))
        {
            std::istringstream linestream(line);
            linestream >> value;

            if (value.find("[") != std::string::npos)
            {
                // linestream >> value;
                LayerType layer = getLayer(value);
                layers.push_back(layer);
            }
            else if (line == "")
            {
                layerProperties.push_back(properties);
                properties.clear();
            }
            else
            {
                // linestream >> value;
                properties.push_back(value);
            }
        }
        layerProperties.push_back(properties);
    }
}

LayerType Config::getLayer(std::string value)
{
    try
    {
        if (value == "[dense]")
            return LayerType::dense;
        else
            throw "Unrecognized layer type: " + value;
    }
    catch (std::string e)
    {
        std::cout << e << std::endl;
    }
}

ActivationFunctionType Config::getActivationFunction(std::string value)
{
    try
    {
        if (value == "relu")
            return ActivationFunctionType::relu;
        else if (value == "sigmoid")
            return ActivationFunctionType::softmax;
        else
            throw "Unrecoknized activation function: " + value;
    }
    catch (std::string e)
    {
        std::cout << e << std::endl;
    }
}

void Config::printConfig()
{
    std::map<LayerType, std::string> mapLayerType;
    mapLayerType[LayerType::dense] = "DenseLayer";

    for (int l = 0; l < layers.size(); l++)
    {
        std::cout << "Layer Type:\n    " << mapLayerType[layers[l]] << std::endl;

        std::cout << "Layer Properties:" << std::endl;

        for (int p = 0; p < layerProperties[l].size(); p++)
        {
            std::cout << "    " << layerProperties[l][p] << std::endl;
        }
        std::cout << std::endl;
    }
}