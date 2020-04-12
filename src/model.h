#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <variant>
#include <Eigen/Dense>

#include "loss.h"
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
    void trainSingle();
    void trainBatch();
    void predictSingle();
    void predictBatch();

    void testForwardPass();

    int batchSize{1};

private:
    // Vector of pointers to neural network layers.
    // Layers can be of multiple layer types, currently only includes DenseLayer
    // std::vector<std::unique_ptr<std::variant<DenseLayer>>> _layers;
    // std::vector<ModelLayer> _layers;
    std::vector<std::variant<DenseLayer>> _layers;
    CategoricalCrossEntropy _loss;
};

#endif