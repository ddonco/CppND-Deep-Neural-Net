#include "loss.h"

Loss::Loss()
{
    _backpassDeltaValues = std::make_unique<Eigen::MatrixXf>();
}

float CategoricalCrossEntropy::forward(Eigen::MatrixXf yPred, Eigen::MatrixXf yTrue)
{
    // number of training samples
    int numSamples = yPred.rows();
    // For categorical labels, calculate probabilities
    if (yTrue.cols() == 1)
    {
        Eigen::MatrixXf yPredArray = Eigen::MatrixXf::Zero(numSamples, 1);
        for (int r = 0; r < numSamples; r++)
        {
            int yTrueCat = yTrue(r, Eigen::all).maxCoeff();
            yPredArray(r, 0) = yPred(r, yTrueCat);
        }
        yPred = yPredArray;
    }

    // Calculate losses
    Eigen::MatrixXf negLogLikelihoods = yPred.array().log();
    negLogLikelihoods = negLogLikelihoods * -1;

    // For on-hot-encoded labels, mask labels with likelihoods
    if (yTrue.cols() == 2)
    {
        negLogLikelihoods = negLogLikelihoods * yTrue;
    }

    // Return overall loss of training data
    return negLogLikelihoods.sum() / numSamples;
}

void CategoricalCrossEntropy::backward(Eigen::MatrixXf backpassDeltaValues, Eigen::MatrixXf yTrue)
{
    // number of training samples
    int numSamples = yTrue.rows();
    int yTrueCat;

    *_backpassDeltaValues = backpassDeltaValues;
    for (int r = 0; r < numSamples; r++)
    {
        yTrueCat = yTrue(r, 0);
        (*_backpassDeltaValues)(r, yTrueCat) = (*_backpassDeltaValues)(r, yTrueCat) - 1;
        (*_backpassDeltaValues) = *_backpassDeltaValues / numSamples;
    }
}