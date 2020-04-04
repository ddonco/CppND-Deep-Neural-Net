#ifndef MODEL_H
#define MODEL_H

#include <variant>
#include "utils.h"

class Model
{
public:
    Model(Config config);
    void train();
    void predict();

private:
    // Vector of pointers to neural network layers.
    // Layers can be of multiple layer types, currently only includes DenseLayer
    std::vector<std::unique_ptr<std::variant<DenseLayer>>> _layers;
};

#endif