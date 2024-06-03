#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "math_utils.h"
#include "linear.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

LinearProjection::LinearProjection(int inputSize, int outputSize, float learningRate, float clipNorm, int seed)
    : learningRate(learningRate), clipNorm(clipNorm) {
    weights = MatrixXd::Zero(inputSize, outputSize);
    MathUtils::initialiseWeights(weights, seed);
    bias = VectorXd::Zero(outputSize);
}

Tensor3D LinearProjection::feedForward(const Tensor3D& input) {
    layerInput = input;
    int batchSize = input.dimension(0);
    int sequenceLength = input.dimension(1);
    int inputSize = input.dimension(2);

    MatrixXd inputMatrix = MathUtils::reshapeToMatrix(input);
    MatrixXd outputMatrix = ((inputMatrix * weights).rowwise() + bias.transpose()).cwiseMax(0);
    layerOutput = MathUtils::reshapeToTensor(outputMatrix, batchSize, sequenceLength, weights.cols());

    return layerOutput;
}

Tensor3D LinearProjection::backPropagation(const Tensor3D& prevError) {
    int batchSize = prevError.dimension(0);
    int sequenceLength = prevError.dimension(1);
    int outputSize = prevError.dimension(2);

    // Calculates the derivative of the ReLU function
    Tensor3D reluDerivative = layerInput.unaryExpr([](float x) { return (x > 0.0f) ? 1.0f : 0.0f; });
    MatrixXd errorMatrix = MathUtils::reshapeToMatrix(prevError);
    MatrixXd reluDerivativeMatrix = MathUtils::reshapeToMatrix(reluDerivative);
    MatrixXd inputErrorMatrix = (errorMatrix.array() * reluDerivativeMatrix.array()).matrix();
    return MathUtils::reshapeToTensor(inputErrorMatrix, batchSize, sequenceLength, weights.rows());
}

void LinearProjection::updateParameters(const Tensor3D& error) {
    MatrixXd inputMatrix = MathUtils::reshapeToMatrix(layerInput);
    MatrixXd errorMatrix = MathUtils::reshapeToMatrix(error);

    MatrixXd weightsGradient = inputMatrix.transpose() * errorMatrix;
    float scaleFactor = MathUtils::matrixClippingFactor(weightsGradient, clipNorm);

    weights -= learningRate * scaleFactor * weightsGradient;
    bias -= learningRate * scaleFactor * errorMatrix.colwise().sum();
}

MatrixXd LinearProjection::getWeights() {
    return weights;
}

Tensor3D LinearProjection::getOutput() {
    return layerOutput;
}