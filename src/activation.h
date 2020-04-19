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
    Eigen::MatrixXf *_input;
    Eigen::MatrixXf *_output;
    Eigen::MatrixXf *_backpassDeltaValues;
};

class Relu
{
public:
    Relu();
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);

    // private:
    Eigen::MatrixXf *_input;
    Eigen::MatrixXf *_output;
    Eigen::MatrixXf *_backpassDeltaValues;
};

class Softmax : public Activation
{
public:
    using Activation::Activation;
    void forward(Eigen::MatrixXf &m);
    void backward(Eigen::MatrixXf &m);
};

#endif