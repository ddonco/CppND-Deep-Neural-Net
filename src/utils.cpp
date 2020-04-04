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
                LayerType layer = getLayer(value);
                layers.push_back(layer);
            }
            else if (line == "")
            {
                layerProperties.push_back(propertiesMap);
                properties.clear();
            }
            else
            {
                std::vector<std::string> props = getProperties(value);
                propertiesMap[props[0]] = props[1];
            }
        }
        layerProperties.push_back(propertiesMap);
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

std::vector<std::string> Config::getProperties(std::string value)
{
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
    return splitString(value, '=');
}

ActivationFunctionType Config::getActivationFunction(std::string value)
{
    try
    {
        value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
        std::vector<std::string> splitValue = splitString(value, '=');
        if (splitValue[1] == "relu")
            return ActivationFunctionType::relu;
        else if (splitValue[1] == "sigmoid")
            return ActivationFunctionType::softmax;
        else
            throw "Unrecoknized activation function: " + value;
    }
    catch (std::string e)
    {
        std::cout << e << std::endl;
    }
}

int Config::getInputsOutputs(std::string value)
{
    try
    {
        value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
        std::vector<std::string> splitValue = splitString(value, '=');
        return std::stoi(splitValue[1]);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
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

        std::map<std::string, std::string> properties = layerProperties[l];
        std::map<std::string, std::string>::iterator it = properties.begin();
        while (it != properties.end())
        {
            std::cout << "    " << it->first << " = " << it->second << std::endl;
            it++;
        }
        std::cout << std::endl;
    }
}

std::vector<std::string> Config::splitString(const std::string &s, char delimiter)
{
    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}