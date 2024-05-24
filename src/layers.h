#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <Eigen/Dense>
#include "attention.h"
#include "linear.h"
#include "math_utils.h"

using namespace std;

class TransformerLayer {
public:
    TransformerLayer(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed);
    void intialiseLayerNormalisationValues(int numNormalisationLayers, int outputSize);
    virtual void updateParameters(const Tensor3D& error) = 0;
    virtual void updateNormalisationParameters(int parameterIndex, const Tensor3D& gammaGradients, const Tensor3D& betaGradients);

protected:
    vector<SelfAttention> selfAttentions;
    vector<LinearProjection> linearProjections1;
    vector<LinearProjection> linearProjections2;
    vector<Tensor3D> gamma;
    vector<Tensor3D> beta;
    int numLayers;
    int numHeads;
    float learningRate;
    float clipNorm;
    const float epsilon = 1e-5; 
};

class Encoder : public TransformerLayer {
public:
    Encoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed);
    Tensor3D feedForward(const Tensor3D& input);
    void updateParameters(const Tensor3D& error) override;
};

class Decoder : public TransformerLayer {
public:
    Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed);
    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& encoderOutput);
    void updateParameters(const Tensor3D& error) override;
    Tensor3D getEncoderError();

private:
    vector<SelfAttention> maskedSelfAttentions;
    LinearProjection outputProjection;
    Tensor3D mask;
    Tensor3D encoderError;
    Tensor3D encoderOutput;
    Tensor3D generateDecoderMask(int inputSize);
};

#endif