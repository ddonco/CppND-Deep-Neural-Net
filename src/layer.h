#ifndef LAYER_H
#define LAYER_H

#include <iostream>
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

protected:
    int _inputs;
    int _outputs;
    std::vector<std::vector<float>> _weights;
    float _bias{0};
    ActivationFunctionType _activation;
};

class DenseLayer : public Layer
{
public:
    DenseLayer(int inputs, int outputs, ActivationFunctionType activation);
    std::vector<std::string> propertiesRequired{"inputs", "outputs", "activation"};
    std::vector<std::string> propertiesOptional{"dropout"};

private:
    float _dropoutRate;
};

#endif