#include "activation.h"

Relu::Relu()
{
    _input = std::make_unique<Eigen::MatrixXf>();
}

void Relu::forward(Eigen::MatrixXf &m)
{
    *_input = Eigen::MatrixXf::Zero(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
        {
            if (m(i, j) > 0)
                (*_input)(i, j) = m(i, j);
        }
    }
}