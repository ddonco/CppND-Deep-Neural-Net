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
        Eigen::MatrixXf yPredLargest = Eigen::MatrixXf::Zero(numSamples, 1);
        for (int r = 0; r < numSamples; r++)
        {
            // yPredLargest(r, 1) = yPred(r, yTrue(r));
        }
    }

    // Calculate losses
    Eigen::MatrixXf negLogLikelihoods = yPred.array().log();
    // negLogLikelihoods = negLogLikelihoods * -1;

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

    *_backpassDeltaValues = backpassDeltaValues;
    for (int r = 0; r < numSamples; r++)
    {
        // (*_backpassDeltaValues)(r, yTrue(r)) = (*_backpassDeltaValues)(r, yTrue(r)) - 1;
        (*_backpassDeltaValues) = *_backpassDeltaValues / numSamples;
    }
}