#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <variant>
#include "utils.h"

// struct ModelLayer
// {
//     LayerType type;
//     std::variant<std::unique_ptr<DenseLayer>> layer;
// };

class Model
{
public:
    Model(Config config);
    void printModel();
    void train();
    void predict();

private:
    // Vector of pointers to neural network layers.
    // Layers can be of multiple layer types, currently only includes DenseLayer
    // std::vector<std::unique_ptr<std::variant<DenseLayer>>> _layers;
    // std::vector<ModelLayer> _layers;
    std::vector<std::variant<DenseLayer>> _layers;
};

#endif