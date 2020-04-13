#ifndef LOSS_H
#define LOSS_H

#include <memory>
#include <iostream>
#include <Eigen/Dense>

class Loss
{
public:
    Loss();

protected:
    std::unique_ptr<Eigen::MatrixXf> _backpassDeltaValues;
};

class CategoricalCrossEntropy : public Loss
{
public:
    using Loss::Loss;
    float forward(Eigen::MatrixXf yPred, Eigen::MatrixXf yTrue);
    void backward(Eigen::MatrixXf backpassDeltaValues, Eigen::MatrixXf yTrue);
};

#endif