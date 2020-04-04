#include "model.h"

Model::Model(Config config)
{
    for (int i = 0; i < config.layers.size(); i++)
    {
        LayerType layer = config.layers[i];
        switch (layer)
        {
        case dense:
        {
            std::vector<std::string> properties = config.layerProperties[i];
            std::unique_ptr<DenseLayer> l = std::make_unique<DenseLayer>();
            _layers.emplace_back(l);
            break;
        }
        default:
            break;
        }
    }
}