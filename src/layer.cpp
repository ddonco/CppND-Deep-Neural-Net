#include "layer.h"
#include "utils.h"

Layer::Layer() {}

Layer::Layer(int inputs, int outputs, ActivationFunctionType activation)
	: _inputs(inputs), _outputs(outputs), _activation(activation)
{
	// _weights = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(outputs, inputs));
	_weights = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(outputs, inputs));
	_bias = std::make_unique<float>(0);
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

Eigen::MatrixXf Layer::forwardPass(Eigen::MatrixXf &input)
{
	std::cout << "Input matrix:\n"
			  << input << std::endl;
	std::cout << "Weight matrix:\n"
			  << *(this->_weights) << std::endl;
	Eigen::MatrixXf output = (input * (*(this->_weights)).transpose()).array() + *(this->_bias);
	return output;
}

DenseLayer::DenseLayer() {}

DenseLayer::DenseLayer(int inputs, int outputs, ActivationFunctionType activation, float dropout = 0.0)
	: Layer(inputs, outputs, activation), _dropoutRate(dropout)
{
}

void DenseLayer::printLayer()
{
	std::map<ActivationFunctionType, std::string> mapActivationType;
	mapActivationType[ActivationFunctionType::relu] = "relu";
	mapActivationType[ActivationFunctionType::softmax] = "softmax";
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	std::cout << "Layer Type:\n    Dense Layer" << std::endl;
	std::cout << "Layer Properties:"
			  << "\n    Inputs: " << this->_inputs
			  << "\n    Outputs: " << this->_outputs
			  << "\n    Activation: " << mapActivationType[this->_activation]
			  << "\n    Weights:\n"
			  << (*_weights).format(CleanFmt)
			  << "\n    Bias: " << *_bias
			  << "\n"
			  << std::endl;
}