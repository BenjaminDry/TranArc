#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include "math_utils.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

class LinearProjection {
public:
    // Constructor, intialises linear layer weights & bias
    LinearProjection(int inputSize, int outputSize, float learningRate, float clipNorm, int seed)
    : learningRate(learningRate), clipNorm(clipNorm) {
        weights = MatrixXd::Zero(inputSize, outputSize);
        MathUtils::initialiseWeights(weights, seed);
        bias = VectorXd::Zero(outputSize);
    }

    // Compute layer output
    Tensor3D feedForward(const Tensor3D& input) {
        layerInput = input;
        int batchSize = input.dimension(0);
        int sequenceLength = input.dimension(1);
        int inputSize = input.dimension(2);

        MatrixXd inputMatrix = MathUtils::reshapeToMatrix(input);
        MatrixXd outputMatrix = MathUtils::sigmoid(inputMatrix * weights + bias.replicate(batchSize * sequenceLength, 1));
        
        layerOutput = MathUtils::reshapeToTensor(outputMatrix, batchSize, sequenceLength, weights.cols());
        return layerOutput;
    }

    // Compute error of the linear layer, based on previous layer error
    Tensor3D backPropagation(const Tensor3D& prevError) {
        int batchSize = prevError.dimension(0);
        int sequenceLength = prevError.dimension(1);
        int outputSize = prevError.dimension(2);

        MatrixXd errorMatrix = MathUtils::reshapeToMatrix(prevError);
        MatrixXd inputErrorMatrix = errorMatrix * weights.transpose();
        return MathUtils::reshapeToTensor(inputErrorMatrix, batchSize, sequenceLength, weights.rows());
    }

    // Projection layer parameter updating, using gradient clipping
    void updateParameters(const Tensor3D& error) {
        MatrixXd inputMatrix = MathUtils::reshapeToMatrix(layerInput);
        MatrixXd errorMatrix = MathUtils::reshapeToMatrix(error);

        MatrixXd weightsGradient = inputMatrix.transpose() * errorMatrix;
        float scaleFactor = MathUtils::matrixClippingFactor(weightsGradient, clipNorm);

        weights -= learningRate * scaleFactor * weightsGradient;
        bias -= learningRate * scaleFactor * errorMatrix.colwise().sum();
    }

    MatrixXd getWeights() {
        return weights;
    }

private:
    // Learning variables
    float learningRate;
    float clipNorm;

    // Adjustable parameters
    MatrixXd weights;
    VectorXd bias;

    // IO Data storage
    Tensor3D layerInput;
    Tensor3D layerOutput;
};