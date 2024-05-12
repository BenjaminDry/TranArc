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

private:
    LinearProjection queryProjection;
    LinearProjection keyProjection;
    LinearProjection valueProjection;
    int numHeads;
    LinearProjection outputProjection;
    void computeQKV(const Tensor3D& input, Tensor3D& queries, Tensor3D& keys, Tensor3D& values, const Tensor3D& mask);
    Tensor3D computeSelfAttention(const Tensor3D& queries, const Tensor3D& keys, const Tensor3D& values);
};

#endif