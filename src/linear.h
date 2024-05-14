#ifndef LINEAR_PROJECTION_H
#define LINEAR_PROJECTION_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include "math_utils.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

class LinearProjection {
public:
    LinearProjection(int inputSize, int outputSize, float learningRate, float clipNorm, int seed);
    Tensor3D feedForward(const Tensor3D& input);
    Tensor3D backPropagation(const Tensor3D& prevError);
    void updateParameters(const Tensor3D& error);
    MatrixXd getWeights();

private:
    float learningRate;
    float clipNorm;
    MatrixXd weights;
    VectorXd bias;
    Tensor3D layerInput;
    Tensor3D layerOutput;
};

#endif