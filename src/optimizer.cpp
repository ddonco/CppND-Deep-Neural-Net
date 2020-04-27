#include "optimizer.h"

StochasticGradientDescent::StochasticGradientDescent(float learningRate)
{
    learningRate = learningRate;
}

void StochasticGradientDescent::updateParams(Layer &layer)
{
    *(layer._weights) = *(layer._weightsDelta) * learningRate;
}