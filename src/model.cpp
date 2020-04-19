#include "model.h"
#include "activation.h"
#include "utils.h"
#include "layer.h"

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

                // std::unique_ptr<DenseLayer> layertest = std::make_unique<DenseLayer>(std::stoi(properties["inputs"]),
                //                                                                  std::stoi(properties["outputs"]),
                //                                                                  actFunction,
                //                                                                  dropout);
                // DenseLayer layer = DenseLayer(std::stoi(properties["inputs"]),
                //                               std::stoi(properties["outputs"]),
                //                               actFunction,
                //                               dropout);

                _layers.emplace_back(DenseLayer(std::stoi(properties["inputs"]),
                                                std::stoi(properties["outputs"]),
                                                batchSize,
                                                actFunction,
                                                dropout));
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
            _activationLayers.emplace_back(Relu());
            break;

        case ActivationFunctionType::softmax:
            _activationLayers.emplace_back(Softmax());

        default:
            break;
        }
    }
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
    // for (auto &layer : _layers)
    for (int l = 0; l < _layers.size(); l++)
    {
        if (auto value = std::get_if<DenseLayer>(&(_layers[l])))
        {
            DenseLayer &v = *value;
            v.printLayer();
        }
    }
}

void Model::testForwardPass(std::unique_ptr<Eigen::MatrixXf> trainX, std::unique_ptr<Eigen::MatrixXf> trainY)
{
    std::cout << "*** Forward Pass Test ***" << std::endl;

    Eigen::MatrixXf layerOut = *trainX; // Eigen::MatrixXf::Constant(1, 4, 2);
    // Eigen::MatrixXf ypred = Eigen::MatrixXf::Constant(1, 1, 2);
    Eigen::MatrixXf ytrue = Eigen::MatrixXf::Constant(1, 1, 2);
    // Eigen::MatrixXf layerOut;

    // auto &layer = _layers[0];

    for (int i = 0; i < _layers.size(); i++)
    {
        if (auto l = std::get_if<DenseLayer>(&(_layers[i])))
        {
            DenseLayer &layer = *l;
            layer.forward(layerOut);
            layerOut = *(layer._output);
        }

        if (auto a = std::get_if<Relu>(&(_activationLayers[i])))
        {
            Relu &activation = *a;
            activation.forward(layerOut);
            layerOut = *(activation._output);
        }
        else if (auto a = std::get_if<Softmax>(&(_activationLayers[i])))
        {
            Softmax &activation = *a;
            activation.forward(layerOut);
            layerOut = *(activation._output);
        }
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
    std::cout << "loss delta address: " << _loss._backpassDeltaValues.get() << std::endl;
    std::cout << "copied delta address: " << &backpassDeltaValues << std::endl;

    for (int i = _layers.size(); i > 0; i--)
    {
        if (auto a = std::get_if<Relu>(&(_activationLayers[i])))
        {
            std::cout << "backward step: " << i << std::endl;
            Relu &activation = *a;
            activation.backward(backpassDeltaValues);
            backpassDeltaValues = *(activation._output);
        }
        else if (auto a = std::get_if<Softmax>(&(_activationLayers[i])))
        {
            Softmax &activation = *a;
            activation.backward(backpassDeltaValues);
            backpassDeltaValues = *(activation._output);
        }

        if (auto l = std::get_if<DenseLayer>(&(_layers[i])))
        {
            DenseLayer &layer = *l;
            layer.backward(backpassDeltaValues);
            backpassDeltaValues = *(layer._output);
        }
    }

    std::cout << "final backward: " << backpassDeltaValues.rows() << ", " << backpassDeltaValues.cols() << "\n"
              << std::endl;
}