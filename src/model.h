#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <variant>
#include <Eigen/Dense>

#include "activation.h"
#include "loss.h"
#include "utils.h"

class Model
{
public:
    Model(Config config);
    void trainSingle();
    void trainBatch();
    void predictSingle();
    void predictBatch();
    Eigen::MatrixXf getPredCategories(Eigen::MatrixXf layerOutput);
    float accuracy(Eigen::MatrixXf yPred, Eigen::MatrixXf yTrue);
    void printModel();

    void testForwardPass(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY);

    int batchSize{1};

private:
    // Vector of pointers to neural network layers.
    // Layers can be of multiple layer types, currently only includes DenseLayer
    std::vector<std::variant<DenseLayer>> _layers;
    std::vector<std::variant<Relu, Softmax>> _activationLayers;
    CategoricalCrossEntropy _loss;
};

#endif