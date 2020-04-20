#include "activation.h"

Activation::Activation()
{
    _input = std::make_unique<Eigen::MatrixXf>();
    _output = std::make_unique<Eigen::MatrixXf>();
    _backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
}

Relu::Relu()
{
    _input = std::make_unique<Eigen::MatrixXf>();
    _output = std::make_unique<Eigen::MatrixXf>();
    _backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
}

void Relu::forward(Eigen::MatrixXf &m)
{
    *_input = m;
    *_output = Eigen::MatrixXf::Zero(m.rows(), m.cols());
    for (int i = 0; i < (*_input).rows(); i++)
    {
        for (int j = 0; j < (*_input).cols(); j++)
        {
            if ((*_input)(i, j) > 0)
                (*_output)(i, j) = m(i, j);
        }
    }
}

void Relu::backward(Eigen::MatrixXf &m)
{
    *_backpassDeltaValues = m;

    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
        {
            if ((*_input)(i, j) <= 0)
            {
                (*_backpassDeltaValues)(i, j) = 0;
            }
        }
    }
}

// Softmax::Softmax()
// {
//     _input = std::make_unique<Eigen::MatrixXf>();
//     _output = std::make_unique<Eigen::MatrixXf>();
//     _backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
// }

void Softmax::forward(Eigen::MatrixXf &m)
{
    *_input = m;
    *_output = Eigen::MatrixXf::Zero(m.rows(), m.cols());
    Eigen::MatrixXf expValues = (m - m.rowwise().maxCoeff()).array().exp();
    Eigen::MatrixXf expValuesRowSum = expValues.rowwise().sum();

    for (int i = 0; i < expValues.rows(); i++)
    {
        for (int j = 0; j < expValues.cols(); j++)
        {
            (*_output)(i, j) = expValues(i, j) / expValuesRowSum(i);
        }
    }
}

void Softmax::backward(Eigen::MatrixXf &m)
{
    *_backpassDeltaValues = m;
}