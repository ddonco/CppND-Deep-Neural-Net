#include "layer.h"

DenseLayer::DenseLayer(int inputs, int outputs, ActivationFunctionType activation)
    : _inputs(inputs),
      _outputs(outputs),
      _activation(activation)
{
}