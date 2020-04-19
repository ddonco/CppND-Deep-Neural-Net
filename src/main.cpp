#include "utils.h"
#include "model.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

// template <typename M>
// M readCsv(const std::string &path)
// {
//     std::ifstream indata;
//     indata.open(path);
//     std::string line;
//     std::vector<float> values;
//     uint rows = 0;
//     while (std::getline(indata, line))
//     {
//         std::stringstream lineStream(line);
//         std::string cell;
//         while (std::getline(lineStream, cell, ','))
//         {
//             values.push_back(std::stod(cell));
//         }
//         ++rows;
//     }
//     return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime>>(values.data(), rows, values.size() / rows);
// }

int main()
{
    std::string configPath = "../config/l3.config";

    std::unique_ptr<Eigen::MatrixXf> trainX = std::make_unique<Eigen::MatrixXf>(readCsv<Eigen::MatrixXf>("../data/X.csv"));
    std::unique_ptr<Eigen::MatrixXf> trainY = std::make_unique<Eigen::MatrixXf>(readCsv<Eigen::MatrixXf>("../data/Y.csv"));
    // Eigen::MatrixXf trainY = readCsv<Eigen::MatrixXf>("../data/Y.csv");
    // std::cout << *trainY << std::endl;

    Config networkConfig;
    networkConfig.readConfig(configPath);
    // networkConfig.printConfig();

    Model model = Model(networkConfig);
    model.printModel();

    model.testForwardPass(std::move(trainX), std::move(trainY));

    // Eigen::MatrixXf mat(2, 2);
    // mat << 1, 2,
    //     3, 4;
    // cout << "Here is mat.sum():       " << mat.sum() << endl;
    // cout << "Here is mat colwise sum: \n"
    //      << mat.colwise().sum() << endl;
    // cout << "colwise sum shape: " << mat.colwise().sum().rows() << ", " << mat.colwise().sum().cols() << endl;
    // cout << "Here is mat rowwise sum: \n"
    //      << mat.rowwise().sum() << endl;
    // cout << "rowwise sum shape: " << mat.rowwise().sum().rows() << ", " << mat.rowwise().sum().cols() << endl;
    // cout << "Here is mat max rowwise: \n"
    //      << mat.rowwise().maxCoeff() << endl;
    // cout << "Here is mat.maxCoeff():  \n"
    //      << mat.maxCoeff() << endl;
    // cout << "Elementwise exp:\n"
    //      << mat.array().exp() << endl;

    return 0;
}
