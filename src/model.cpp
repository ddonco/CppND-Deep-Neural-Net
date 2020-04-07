#include "model.h"
#include "activation.h"
#include "utils.h"
#include "layer.h"

Model::Model(Config config)
{
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

                ActivationFunctionType actFunction = Utils::parseActivationFunction(properties["activation"]);

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
    }
}

void Model::printModel()
{
    for (auto &layer : _layers)
    {
        if (auto value = std::get_if<DenseLayer>(&layer))
        {
            DenseLayer &v = *value;
            v.printLayer();
        }
    }
}

void Model::testForwardPass()
{
    std::cout << "*** Forward Pass Test ***" << std::endl;

    Eigen::MatrixXf m = Eigen::MatrixXf::Constant(1, 4, 1);
    // std::cout << "Input matrix:\n"
    //           << m << std::endl;
    auto &layer = _layers[0];

    if (auto value = std::get_if<DenseLayer>(&layer))
    {
        DenseLayer &v = *value;
        Eigen::MatrixXf o = v.forwardPass(m);
        std::cout << "Output matrix:\n"
                  << o << std::endl;
    }
}