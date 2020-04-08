#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <memory>
#include <Eigen/Dense>

enum ActivationFunctionType
{
    relu,
    softmax
};

class Relu
{
public:
    void forward(Eigen::MatrixXf &m);
    void backward();

private:
    std::unique_ptr<Eigen::MatrixXf> _input;
};

class Softmax
{
public:
    void forward(Eigen::MatrixXf &m);
    void backward();

private:
    std::unique_ptr<Eigen::MatrixXf> _input;
};

#endif