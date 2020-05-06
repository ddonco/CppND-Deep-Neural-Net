#ifndef MODEL_H
#define MODEL_H

#include <memory>
#include <mutex>
#include <variant>
#include <Eigen/Dense>

#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "utils.h"

class Model
{
public:
    Model(Config config);
    float accuracy(Eigen::MatrixXf *yPred, Eigen::MatrixXf *yTrue);
    Eigen::MatrixXf *getPredCategories(Eigen::MatrixXf *layerOutput);
    void loadWeights(const std::string &weightsPath);
    void predictBatch();
    void predictSingle();
    void trainBatch();
    void printModel();
    void saveWeights(const std::string &weightsPath);
    void testForwardPass(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY);
    void train(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY);
    void trainSingle();

    int batchSize{1};

private:
    // Vector of pointers to neural network layers.
    // Layers can be of multiple layer types, currently only includes DenseLayer
    std::vector<Layer *> _layers;
    std::vector<Activation *> _activationLayers;
    CategoricalCrossEntropy _loss;
    StochasticGradientDescent _optimizer;
};

#endif