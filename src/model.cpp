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

void Model::testForwardPass()
{
    std::cout << "*** Forward Pass Test ***" << std::endl;

    Eigen::MatrixXf layerOut = Eigen::MatrixXf::Constant(1, 4, 1);
    Eigen::MatrixXf ypred = Eigen::MatrixXf::Constant(1, 1, 1);
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

    float lossValue = _loss.forward(layerOut, ytrue);
    std::cout << "Loss value: " << lossValue << "\n"
              << std::endl;
}