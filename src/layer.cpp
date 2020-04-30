#include "layer.h"
#include "model.h"
#include "utils.h"

Layer::Layer(int inputs, int outputs, int batchSize, ActivationFunctionType activation)
	: _inputs(inputs), _outputs(outputs), batchSize(batchSize), _activation(activation)
{
	// Randomize weights matrix initialization
	// srand((unsigned int)time(0));

	_input = std::make_unique<Eigen::MatrixXf>();
	_output = std::make_unique<Eigen::MatrixXf>();
	_weights = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(outputs, inputs));
	*_weights *= 0.01;
	_weightsDelta = std::make_unique<Eigen::MatrixXf>();
	_backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
	_bias = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(batchSize, outputs));
	_biasDelta = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(batchSize, outputs));
}

void Layer::printLayer()
{
	std::map<ActivationFunctionType, std::string> mapActivationType;
	mapActivationType[ActivationFunctionType::relu] = "relu";
	mapActivationType[ActivationFunctionType::softmax] = "softmax";
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	std::cout << "Layer Type:\n    Layer" << std::endl;
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
	// std::cout << "Input matrix:\n"
	// 		  << m.rows() << ", " << m.cols() << std::endl;
	// std::cout << "Weight matrix:\n"
	// 		  << (*_weights).rows() << ", " << (*_weights).cols() << std::endl;

	// Save input values
	*_input = m;

	// Calculate forward pass
	*_output = (m * (*_weights).transpose());

	for (int row = 0; row < (*_output).rows(); row++)
	{
		for (int col = 0; col < (*_output).cols(); col++)
		{
			(*_output)(row, col) += (*_bias)(0, col);
		}
	}

	// std::cout << "Output matrix:\n"
	// 		  << (*_output).rows() << ", " << (*_output).cols() << std::endl;
	// std::cout << "Output matrix:\n"
	// 		  << *_output << "\n"
	// 		  << std::endl;
}

void Layer::backward(Eigen::MatrixXf &m)
{
	// Calculate gradient of the weights
	// std::cout << "weights delta matrix: "
	// 		  << m.rows() << ", " << m.cols()
	// 		  << " * "
	// 		  << (*_input).rows() << ", " << (*_input).cols() << " = " << std::endl;

	*_weightsDelta = m.transpose() * *_input;
	// std::cout << (*_weightsDelta).rows() << ", " << (*_weightsDelta).cols()
	// 		  << "\n"
	// 		  << std::endl;

	*_biasDelta = m.colwise().sum();

	// Calculate gradient of the backward pass values
	// std::cout << "values backdelta matrix: "
	// 		  << m.rows() << ", " << m.cols()
	// 		  << " * "
	// 		  << (*_weights).rows() << ", " << (*_weights).cols()
	// 		  << " = " << std::endl;
	*_backpassDeltaValues = m * (*_weights); // need to verify its not weights.transpose()
											 // std::cout << "backpass delta values: " << (*_backpassDeltaValues).rows() << ", " << (*_backpassDeltaValues).cols()
											 // 		  << "\n"
											 // 		  << std::endl;
}

DenseLayer::DenseLayer(int inputs, int outputs, int batchSize, ActivationFunctionType activation, float dropout = 0.0)
	: Layer(inputs, outputs, batchSize, activation), _dropoutRate(dropout)
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