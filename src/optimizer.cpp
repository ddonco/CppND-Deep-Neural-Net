#include "optimizer.h"

StochasticGradientDescent::StochasticGradientDescent(float learningRate)
{
    learningRate = learningRate;
}

void StochasticGradientDescent::updateParams(Layer *layer)
{
    std::unique_lock<std::mutex> lck(_mutex);

    std::cout << "weights dims: " << (*layer)._weights->rows() << ", " << (*layer)._weights->cols()
              << std::endl;
    std::cout << "weightDelta dims: " << (*layer)._weightsDelta->rows() << ", " << (*layer)._weightsDelta->cols()
              << std::endl;

    *((*layer)._weights) -= *((*layer)._weightsDelta) * learningRate;
    lck.unlock();
}