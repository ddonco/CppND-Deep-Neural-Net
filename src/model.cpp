#include "model.h"
#include "activation.h"
#include "optimizer.h"
#include "layer.h"
#include "utils.h"

Model::Model(Config config)
{
    batchSize = 1;
    ActivationFunctionType actFunction;
    for (int i = 0; i < config.layers.size(); i++)
    {
        LayerType layerType = config.layers[i];
        switch (layerType)
        {
        case dense:
        {
            try
            {
                std::map<std::string, std::string> properties = config.layerProperties[i];

                actFunction = Utils::parseActivationFunction(properties["activation"]);

                float dropout = 0.0;
                if (properties.count("dropout"))
                    dropout = std::stof(properties["dropout"]);

                DenseLayer *layer = new DenseLayer(std::stoi(properties["inputs"]),
                                                   std::stoi(properties["outputs"]),
                                                   batchSize,
                                                   actFunction,
                                                   dropout);
                _layers.emplace_back(layer);
                break;
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << std::endl;
            }
        }
        default:
            break;
        }

        switch (actFunction)
        {
        case ActivationFunctionType::relu:
        {
            Relu *relu = new Relu();
            _activationLayers.emplace_back(relu);
            break;
        }

        case ActivationFunctionType::softmax:
        {
            Softmax *softmax = new Softmax();
            _activationLayers.emplace_back(softmax);
            break;
        }

        default:
        {
            break;
        }
        }
    }

    CategoricalCrossEntropy _loss();
    StochasticGradientDescent _optimizer(1);
}

Eigen::MatrixXf Model::getPredCategories(Eigen::MatrixXf layerOutput)
{
    Eigen::MatrixXf predictionScores = layerOutput.rowwise().maxCoeff();
    Eigen::MatrixXf predictionCategories = Eigen::MatrixXf::Zero(predictionScores.rows(), predictionScores.cols());
    Eigen::MatrixXf::Index maxIndex;
    for (int i = 0; i < predictionScores.rows(); i++)
    {
        for (int cat = 0; cat < layerOutput.cols(); cat++)
        {
            if (layerOutput(i, cat) == predictionScores(i, 0))
            {
                predictionCategories(i, 0) = cat;
            }
        }
    }
    return predictionCategories;
}

float Model::accuracy(Eigen::MatrixXf yPred, Eigen::MatrixXf yTrue)
{
    float accuracy = 0;
    for (int i = 0; i < yTrue.rows(); i++)
    {
        if (yTrue(i, 0) == yPred(i, 0))
            accuracy++;
    }
    return accuracy / yTrue.rows();
}

void Model::printModel()
{
    for (int l = 0; l < _layers.size(); l++)
    {
        Layer *layer = _layers[l];
        layer->printLayer();
    }
}

void Model::testForwardPass(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY)
{
    std::cout << "*** Forward Pass Test ***" << std::endl;

    Eigen::MatrixXf layerOut = *trainX;

    for (int l = 0; l < _layers.size(); l++)
    {
        Layer *layer = _layers[l];
        layer->forward(layerOut);
        layerOut = *(layer->_output);

        Activation *activation = _activationLayers[l];
        activation->forward(layerOut);
        layerOut = *(activation->_output);
    }

    float loss = _loss.forward(layerOut, *trainY);
    std::cout << "Loss value: " << loss << "\n"
              << std::endl;

    Eigen::MatrixXf yPred = getPredCategories(layerOut);

    std::cout << "final output: " << layerOut.rows() << ", " << layerOut.cols() << std::endl;
    std::cout << "predictions: " << yPred.rows() << ", " << yPred.cols() << "\n"
              << std::endl;

    std::cout << "accuracy: " << accuracy(yPred, *trainY) << "\n"
              << std::endl;

    _loss.backward(layerOut, *trainY);
    Eigen::MatrixXf backpassDeltaValues = *(_loss._backpassDeltaValues);
    std::cout << "loss backward: " << backpassDeltaValues.rows() << ", " << backpassDeltaValues.cols() << "\n"
              << std::endl;

    for (int l = _layers.size() - 1; l >= 0; l--)
    {
        Activation *activation = _activationLayers[l];
        activation->backward(backpassDeltaValues);
        backpassDeltaValues = *(activation->_backpassDeltaValues);

        Layer *layer = _layers[l];
        layer->backward(backpassDeltaValues);
        backpassDeltaValues = *(layer->_backpassDeltaValues);
    }

    std::cout << "final backward: " << backpassDeltaValues.rows() << ", " << backpassDeltaValues.cols() << "\n"
              << std::endl;
    std::cout << "final backward: " << backpassDeltaValues << "\n"
              << std::endl;

    for (int l = 0; l < _layers.size(); l++)
    {
        // Update layer parameters using optimizer
        Layer *layer = _layers[l];

        std::cout << "weights before:\n"
                  << *((*layer)._weights) << "\n"
                  << std::endl;

        _optimizer.updateParams(layer);

        std::cout << "weights after:\n"
                  << *((*layer)._weights) << "\n"
                  << std::endl;
    }
}

void Model::train(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY)
{
    std::cout << "*** Training ***" << std::endl;

    for (int iter = 0; iter < 300; iter++)
    {
        Eigen::MatrixXf layerOut = *trainX;

        for (int l = 0; l < _layers.size(); l++)
        {
            Layer *layer = _layers[l];
            layer->forward(layerOut);
            layerOut = *(layer->_output);

            Activation *activation = _activationLayers[l];
            activation->forward(layerOut);
            layerOut = *(activation->_output);
        }

        float loss = _loss.forward(layerOut, *trainY);

        Eigen::MatrixXf yPred = getPredCategories(layerOut);

        if (iter % 10 == 0)
            std::cout << "Iteration: " << iter << ", Loss: " << loss << ", Accuracy: "
                      << accuracy(yPred, *trainY) << "\n"
                      << std::endl;

        _loss.backward(layerOut, *trainY);
        Eigen::MatrixXf backpassDeltaValues = *(_loss._backpassDeltaValues);

        for (int l = _layers.size() - 1; l >= 0; l--)
        {
            Activation *activation = _activationLayers[l];
            activation->backward(backpassDeltaValues);
            backpassDeltaValues = *(activation->_backpassDeltaValues);

            Layer *layer = _layers[l];
            layer->backward(backpassDeltaValues);
            backpassDeltaValues = *(layer->_backpassDeltaValues);
        }

        for (int l = 0; l < _layers.size(); l++)
        {
            // Update layer parameters using optimizer
            Layer *layer = _layers[l];
            _optimizer.updateParams(layer);
        }
    }
}