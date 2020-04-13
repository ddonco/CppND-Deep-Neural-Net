#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <memory>
#include <iostream>
#include <Eigen/Dense>

enum ActivationFunctionType
{
    relu,
    softmax
};

class Activation
{
public:
    Activation();
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);

    // protected:
    std::unique_ptr<Eigen::MatrixXf> _input;
    std::unique_ptr<Eigen::MatrixXf> _output;
    std::unique_ptr<Eigen::MatrixXf> _backpassDeltaValues;
};

class Relu
{
public:
    Relu();
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);

    // private:
    std::unique_ptr<Eigen::MatrixXf> _input;
    std::unique_ptr<Eigen::MatrixXf> _output;
    std::unique_ptr<Eigen::MatrixXf> _backpassDeltaValues;
};

class Softmax : public Activation
{
public:
    using Activation::Activation;
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);
};

#endif