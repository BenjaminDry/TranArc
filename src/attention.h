#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "linear.h"

using namespace Eigen;

class SelfAttention {
public:
    SelfAttention(int inputSize, int numHeads, float learningRate, float clipNorm, int scalingFactor, int seed);
    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& mask = Tensor3D());
    void backPropagation(const Tensor3D& prevError);
    Tensor3D getLayerOutput();

private:
    void computeQKV();
    Tensor3D computeSelfAttention(const Tensor3D& mask);
    LinearProjection queryProjection;
    LinearProjection keyProjection;
    LinearProjection valueProjection;
    Tensor3D queries;
    Tensor3D keys;
    Tensor3D values;
    vector<Tensor3D> splitQueries;
    vector<Tensor3D> splitKeys;
    vector<Tensor3D> splitValues;
    Tensor3D layerInput;
    Tensor3D layerOutput;
    int numHeads;
    int scalingFactor;
    LinearProjection outputProjection;
};

#endif