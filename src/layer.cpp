#include "layer.h"

Layer::Layer(int inputs, int outputs, ActivationFunctionType activation)
	: _inputs(inputs), _outputs(outputs), _activation(activation)
{
}

DenseLayer::DenseLayer(int inputs, int outputs, ActivationFunctionType activation)
	: Layer(inputs, outputs, activation)
{
}