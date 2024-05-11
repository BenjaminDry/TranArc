#include <vector>
#include "attention.h"
#include "linear.h"
#include "math_utils.h"

using namespace std;

class TransformerLayer {
public:
    TransformerLayer(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : numLayers(numLayers), numHeads(numHeads), learningRate(learningRate), clipNorm(clipNorm) {
        for (int i = 0; i < numLayers; ++i) {
            selfAttentions.push_back(SelfAttention(inputSize, numHeads));
            linearProjections.push_back(LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
        }
    }

    virtual Tensor3D feedForward(const Tensor3D& input) = 0;

protected:
    vector<SelfAttention> selfAttentions;
    vector<LinearProjection> linearProjections;

    int numLayers;
    int numHeads;
    float learningRate;
    float clipNorm;
};

class Encoder : public TransformerLayer {
public:
    Encoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed) {}

    Tensor3D feedForward(const Tensor3D& input) override {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {

            // Encoder specfic feed forward logic

            output = selfAttentions[i].feedForward(output);
            output = linearProjections[i].feedForward(output);
        }

        return output;
    }
};

class Decoder : public TransformerLayer {
public:
    Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed) {}

    Tensor3D feedForward(const Tensor3D& input) override {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {

            // Decoder specfic feed forward logic

            output = selfAttentions[i].feedForward(output);
            output = linearProjections[i].feedForward(output);
        }
        
        return output;
    }

};