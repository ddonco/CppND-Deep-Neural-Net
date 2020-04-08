#include "layer.h"
#include "utils.h"

Layer::Layer() {}

Layer::Layer(int inputs, int outputs, ActivationFunctionType activation)
	: _inputs(inputs), _outputs(outputs), _activation(activation)
{
	_input = std::make_unique<Eigen::MatrixXf>();
	_output = std::make_unique<Eigen::MatrixXf>();
	_weights = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(outputs, inputs));
	_weightsDelta = std::make_unique<Eigen::MatrixXf>();
	_backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
	_bias = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(1, 1));
	_biasDelta = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(1, 1));
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

void Layer::forward(Eigen::MatrixXf &m)
{
	std::cout << "Input matrix:\n"
			  << m << std::endl;
	std::cout << "Weight matrix:\n"
			  << *_weights << std::endl;
	*_input = m;
	// TODO: adding bias as a matrix to a matrix of different dimension not working
	*_output = (m * (*_weights).transpose()).colwise() + (*_bias).array();
	std::cout << "Weight matrix after transpose:\n"
			  << *_weights << std::endl;
	std::cout << "Output matrix:\n"
			  << *_output << std::endl;
}

void Layer::backward(Eigen::MatrixXf &m)
{
	// Calculate gradient of the weights
	*_weightsDelta = (*_input).transpose() * m;

	*_biasDelta = (*_backpassDeltaValues).rowwise().sum();

	// Calculate gradient of the backward pass values
	*_backpassDeltaValues = m * (*_weights).transpose();
}

DenseLayer::DenseLayer()
{
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
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	std::cout << "Layer Type:\n    Dense Layer" << std::endl;
	std::cout << "Layer Properties:"
			  << "\n    Inputs: " << _inputs
			  << "\n    Outputs: " << _outputs
			  << "\n    Activation: " << mapActivationType[_activation]
			  << "\n    Weights:\n"
			  << (*_weights).format(CleanFmt)
			  << "\n    Bias: " << *_bias
			  << "\n"
			  << std::endl;
}