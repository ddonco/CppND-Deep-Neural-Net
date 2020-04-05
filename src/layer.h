#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "activation.h"

enum LayerType
{
    dense
};

class Layer
{
public:
    Layer(int inputs, int outputs, ActivationFunctionType activation);
    void setRequiredProperties(std::map<std::string, std::string> properties);

    std::vector<std::string> propertiesRequired{"inputs", "outputs", "activation"};

protected:
    int _inputs;
    int _outputs;
    ActivationFunctionType _activation;
    bool _inputsSet{false};
    bool _outputsSet{false};
    bool _actiationSet{false};
    std::vector<std::vector<float>> _weights;
    float _bias{0};
};

class DenseLayer : public Layer
{
public:
    DenseLayer(int inputs, int outputs, ActivationFunctionType activation, float dropout);
    void printLayer();

    LayerType type{LayerType::dense};
    std::vector<std::string> propertiesOptional{"dropout"};

private:
    float _dropoutRate;
};

#endif