#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>

#include "layer.h"

class StochasticGradientDescent
{
public:
    StochasticGradientDescent(float learningRate);
    StochasticGradientDescent() {}
    void updateParams(Layer &layer);

    float learningRate{1};
};

#endif