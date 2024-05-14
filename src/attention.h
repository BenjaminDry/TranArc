#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "linear.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

class SelfAttention {
public:
    SelfAttention(int inputSize, int numHeads, float learningRate, float clipNorm, int seed);
    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& mask);
    void backPropagation(const Tensor3D& prevError);
    Tensor3D getLayerOutput();

private:
    LinearProjection queryProjection;
    LinearProjection keyProjection;
    LinearProjection valueProjection;
    Tensor3D queries;
    Tensor3D keys;
    Tensor3D values;
    int numHeads;
    LinearProjection outputProjection;
    Tensor3D layerInput;
    Tensor3D layerOutput;
    void computeQKV();
    Tensor3D computeSelfAttention(const Tensor3D& mask);
};

#endif