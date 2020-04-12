#include "utils.h"
#include "model.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main()
{
    std::string configPath = "../config/l3.config";

    Config networkConfig;
    networkConfig.readConfig(configPath);
    // networkConfig.printConfig();

    Model model = Model(networkConfig);
    model.printModel();

    model.testForwardPass();

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
