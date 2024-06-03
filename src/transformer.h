#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "layers.h"
#include "math_utils.h"

using Tensor3D = Eigen::Tensor<float, 3>;
using namespace std;

class TransformerModel {
public:
    TransformerModel(Encoder& encoderPtr, Decoder& decoderPtr);
    virtual ~TransformerModel();
    void train(const vector<Tensor3D>& inputSeq, vector<Tensor3D>& targetSeq, int epochs);
    Tensor3D predict(const Tensor3D& input);

private:
    Encoder encoder;
    Decoder decoder;
};

#endif
