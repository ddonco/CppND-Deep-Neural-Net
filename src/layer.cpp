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
	_weights = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(inputs, outputs));
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

void Layer::forward(Eigen::MatrixXf *m)
{
	// Save input values
	*_input = *m;

	// std::cout << "Input matrix:\n"
	// 		  << (*_input).rows() << ", " << (*_input).cols() << std::endl;
	// std::cout << "Weight matrix:\n"
	// 		  << (*_weights).rows() << ", " << (*_weights).cols() << std::endl;

	// Calculate forward pass
	*_output = *m * *_weights;

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

void Layer::backward(Eigen::MatrixXf *m)
{
	// Calculate gradient of the weights
	// std::cout << "weights delta matrix: "
	// 		  << (*_input).rows() << ", " << (*_input).cols()
	// 		  << " * "
	// 		  << m.rows() << ", " << m.cols() << " = " << std::endl;

	// std::cout << "input:\n"
	// 		  << *_input << "\n"
	// 		  << std::endl;
	// std::cout << "m:\n"
	// 		  << m << "\n"
	// 		  << std::endl;
	*_weightsDelta = (*_input).transpose() * *m;
	// std::cout << (*_weightsDelta).rows() << ", " << (*_weightsDelta).cols()
	// 		  << "\n"
	// 		  << std::endl;

	*_biasDelta = (*m).colwise().sum();
	// std::cout << "bias delta: " << (*_biasDelta).rows() << ", " << (*_biasDelta).cols() << std::endl;

	// Calculate gradient of the backward pass values
	// std::cout << "values backdelta matrix: "
	// 		  << m.rows() << ", " << m.cols()
	// 		  << " * "
	// 		  << (*_weights).rows() << ", " << (*_weights).cols()
	// 		  << " = " << std::endl;
	*_backpassDeltaValues = *m * (*_weights).transpose(); // need to verify its not weights.transpose()
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
			  //   << "\n    Weights:\n"
			  //   << (*_weights).format(CleanFmt)
			  //   << "\n    Bias: " << *_bias
			  << "\n"
			  << std::endl;
}