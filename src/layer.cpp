#include "layer.h"
#include "utils.h"

Layer::Layer(int inputs, int outputs, ActivationFunctionType activation)
	: _inputs(inputs), _outputs(outputs), _activation(activation)
{
}

void Layer::setRequiredProperties(std::map<std::string, std::string> properties)
{
	for (std::string item : propertiesRequired)
	{
		try
		{
			if (item == "inputs")
			{
				_inputs = std::stoi(properties[item]);
				_inputsSet = true;
			}
			if (item == "outputs")
			{
				_outputs = std::stoi(properties[item]);
				_outputsSet = true;
			}
			if (item == "activation")
			{
				_activation = Utils::parseActivationFunction(properties[item]);
				_actiationSet = true;
			}
		}
		catch (const std::exception &ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}

	try
	{
		if (!_inputsSet)
			throw "Inputs property not set.";
		if (!_outputsSet)
			throw "Outputs property not set.";
		if (!_actiationSet)
			throw "Activation property not set.";
	}
	catch (std::string e)
	{
		std::cout << e << std::endl;
	}
}

DenseLayer::DenseLayer(int inputs, int outputs, ActivationFunctionType activation, float dropout = 0.0)
	: Layer(inputs, outputs, activation), _dropoutRate(dropout)
{
}

void DenseLayer::printLayer()
{
	std::map<ActivationFunctionType, std::string> mapActivationType;
	mapActivationType[ActivationFunctionType::relu] = "relu";
	mapActivationType[ActivationFunctionType::softmax] = "softmax";

	std::cout << "Layer Type:\n    Dense Layer" << std::endl;
	std::cout << "Layer Properties:"
			  << "\n    Inputs: " << this->_inputs
			  << "\n    Outputs: " << this->_outputs
			  << "\n    Activation: " << mapActivationType[this->_activation]
			  << std::endl;
}