#include "optimizer.h"
SGD::SGD(float learningRate) : learningRate(learningRate) {}

void SGD::updateParams(Layer &layer)
{
    *(layer._weights) = *(layer._weightsDelta) * learningRate;
}