#include "utils.h"
#include "model.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc, char *argv[])
{
    std::string mode, config, weights;
    std::string trainXPath, trainYPath;
    std::string testXPath;
    if (argc < 5)
    {
        std::cout << "Usage: [train/test] [config path] [weights path] <option(s)>\n"
                  << "**Note: Train X and Train Y data must be passed (in that order) if mode is 'train'.\n"
                  << "Test X data must be passed if mode is 'test'.**"
                  << "Options:\n"
                  << "    <train X data path if mode is train>\n"
                  << "    <train Y data path if mode is train>\n"
                  << "    <test X data path if mode is test>\n"
                  << std::endl;
        // return 1;
    }
    std::cout << "Arguments:" << std::endl;
    for (int i = 0; i < argc; i++)
    {
        std::cout << i << ": " << argv[i] << std::endl;
    }

    mode = argv[1];
    config = argv[2];
    weights = argv[3];
    if (mode == "train")
    {
        if (argc < 6)
        {
            // return 1;
        }
        trainXPath = argv[4];
        trainYPath = argv[5];
    }
    else if (mode == "test")
    {
        testXPath = argv[4];
    }
    else
    {
        std::cout << "Unsupported argument in position [1]: " << mode << std::endl;
    }
    std::cout << "Arguments:" << std::endl;
    for (int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << std::endl;
    }

    std::string configPath = "../config/l3.config";

    std::unique_ptr<Eigen::MatrixXf> trainX = std::make_unique<Eigen::MatrixXf>(readCsv<Eigen::MatrixXf>("../data/X.csv"));
    std::unique_ptr<Eigen::MatrixXf> trainY = std::make_unique<Eigen::MatrixXf>(readCsv<Eigen::MatrixXf>("../data/Y.csv"));

    // std::unique_ptr<Eigen::MatrixXf> trainX = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(4, 2));
    // *trainX << 1, 2, 3, 4, 5, 6, 7, 8;
    // std::unique_ptr<Eigen::MatrixXf> trainY = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(4, 1));

    // Eigen::MatrixXf trainX = Eigen::MatrixXf::Zero(4, 2);
    // trainX << 1, 2, 3, 4, 5, 6, 7, 8;
    // Eigen::MatrixXf trainY = Eigen::MatrixXf::Ones(4, 2);

    Config networkConfig;
    networkConfig.readConfig(configPath);

    Model model = Model(networkConfig);
    model.printModel();
    model.saveWeights(weights);

    // model.testForwardPass(std::move(trainX), std::move(trainY));
    // model.train(std::move(trainX), std::move(trainY));

    return 0;
}
