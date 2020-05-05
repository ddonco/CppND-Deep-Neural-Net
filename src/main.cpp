#include "utils.h"
#include "model.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc, char *argv[])
{
    std::string mode, config, weights;
    std::string trainX, trainY;
    std::string testX;
    if (argc < 3)
    {
    }

    mode = argv[0];
    config = argv[1];
    weights = argv[2];
    if (mode == "train")
    {
        trainX = argv[3];
        trainY = argv[4];
    }
    else if (mode == "test")
    {
        testX = argv[3];
    }
    else
    {
        std::cout << "Unsupported argument in position [0]: " << mode << std::endl;
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
    // networkConfig.printConfig();

    Model model = Model(networkConfig);
    model.printModel();

    // model.testForwardPass(std::move(trainX), std::move(trainY));
    // model.train(std::move(trainX), std::move(trainY));

    Eigen::MatrixXf *a = new Eigen::MatrixXf(64, 64);
    *a = Eigen::MatrixXf::Random(64, 64);
    Eigen::MatrixXf *b = new Eigen::MatrixXf(64, 64);
    *b = Eigen::MatrixXf::Random(64, 64);

    for (int i = 0; i < 10001; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            Eigen::MatrixXf c = *a * *b;
            for (int row = 0; row < (*a).rows(); row++)
            {
                for (int col = 0; col < (*a).cols(); col++)
                {
                    c(row, col) = (*a)(row, col) + (*b)(row, col);
                }
            }
            for (int row = 0; row < (*b).rows(); row++)
            {
                for (int col = 0; col < (*b).cols(); col++)
                {
                    c(row, col) = (*a)(row, col) + (*b)(row, col);
                }
            }
        }
        if (i % 100 == 0)
            std::cout << "iteration: " << i << std::endl;
    }

    return 0;
}
