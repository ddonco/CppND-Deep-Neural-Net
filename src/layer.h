#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <string>

#include "activation.h"

enum LayerType
{
    dense
};

class DenseLayer
{
public:
    DenseLayer(int inputs, int outputs, ActivationFunctionType activation);

private:
    int _inputs;
    int _outputs;
    ActivationFunctionType _activation;
};

#endif