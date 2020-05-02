#include "optimizer.h"

StochasticGradientDescent::StochasticGradientDescent(float learningRate)
{
    learningRate = learningRate;
}

void StochasticGradientDescent::updateParams(Layer *layer)
{
    std::unique_lock<std::mutex> lck(_mutex);

    // std::cout << "weights dims: " << (*layer)._weights->rows() << ", " << (*layer)._weights->cols()
    //           << std::endl;
    // std::cout << "weightDelta dims: " << (*layer)._weightsDelta->rows() << ", " << (*layer)._weightsDelta->cols()
    //           << std::endl;
    // std::cout << layer->_weights.get() << std::endl;
    // std::cout << "weightDelta: " << *(layer->_weightsDelta) << "\n"
    //           << std::endl;
    // std::cout << "weight: " << *(layer->_weights) << "\n"
    //           << std::endl;

    *(layer->_weights) -= (*(layer->_weightsDelta) * learningRate);
    *(layer->_bias) -= (*(layer->_biasDelta) * learningRate);

    // std::cout << "weightDelta after: " << *(layer->_weightsDelta) << "\n"
    //           << std::endl;
    // std::cout << "weights change: " << *((*layer)._weightsDelta) * learningRate << std::endl;
    // std::cout << "bias change: " << *((*layer)._biasDelta) * learningRate << std::endl;
    lck.unlock();
}