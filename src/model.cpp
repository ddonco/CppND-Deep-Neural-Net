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

                // std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>(std::stoi(properties["inputs"]),
                //                                                                  std::stoi(properties["outputs"]),
                //                                                                  actFunction,
                //                                                                  dropout);
                DenseLayer layer = DenseLayer(std::stoi(properties["inputs"]),
                                              std::stoi(properties["outputs"]),
                                              actFunction,
                                              dropout);

                // std::unique_ptr<ModelLayer> modelLayer = std::make_unique<ModelLayer>();
                // modelLayer->type = layerType;
                // modelLayer->layer = layer;
                ModelLayer modelLayer = {layerType, layer};
                _layers.push_back(modelLayer);
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
    for (auto const &layer : _layers)
    {
        if (layer.type == LayerType::dense)
        {
            std::any_cast<DenseLayer>(layer.layer).printLayer();
        }
    }
}