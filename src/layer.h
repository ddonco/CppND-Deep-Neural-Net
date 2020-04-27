#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <Eigen/Dense>

#include "activation.h"

enum LayerType
{
    dense
};

class Layer
{
public:
    Layer(int inputs, int outputs, int batchSize, ActivationFunctionType activation);
    virtual void printLayer();
    void setRequiredProperties(std::map<std::string, std::string> properties);
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);

    std::vector<std::string> propertiesRequired{"inputs", "outputs", "activation"};
    int batchSize;

    int _inputs;
    int _outputs;
    ActivationFunctionType _activation;
    std::unique_ptr<std::variant<Relu, Softmax>> _activationFunction;
    bool _inputsSet{false};
    bool _outputsSet{false};
    bool _actiationSet{false};
    std::unique_ptr<Eigen::MatrixXf> _input;
    std::unique_ptr<Eigen::MatrixXf> _output;
    std::unique_ptr<Eigen::MatrixXf> _weights;
    std::unique_ptr<Eigen::MatrixXf> _weightsDelta;
    std::unique_ptr<Eigen::MatrixXf> _backpassDeltaValues;
    std::unique_ptr<Eigen::MatrixXf> _bias;
    std::unique_ptr<Eigen::MatrixXf> _biasDelta;
};

class DenseLayer : public Layer
{
public:
    DenseLayer(int inputs, int outputs, int batchSize, ActivationFunctionType activation, float dropout);
    void printLayer();

    LayerType type{LayerType::dense};
    std::vector<std::string> propertiesOptional{"dropout"};

private:
    float _dropoutRate;
};

#endif