#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>

#include "layer.h"

class SGD
{
public:
    SGD(float learningRate);
    void updateParams(Layer &layer);

    float learningRate;
};

#endif