#include <vector>
#include "attention.h"
#include "linear.h"
#include "math_utils.h"

using namespace std;

// Parent class for encoder-decoder layers
class TransformerLayer {
public:
    // Default constructor for each network division, intialises the self attention and linear projection layers
    TransformerLayer(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : numLayers(numLayers), numHeads(numHeads), learningRate(learningRate), clipNorm(clipNorm) {
        for (int i = 0; i < numLayers; ++i) {
            selfAttentions.push_back(SelfAttention(inputSize, numHeads));
            linearProjections.push_back(LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
        }
    }

    // Feed data through the layers
    virtual Tensor3D feedForward(const Tensor3D& input) = 0;

    // Update the parameters of the layers
    virtual void updateParameters() = 0;

    // Update the gamma and beta tensor objects
    virtual void updateNormalisationParameters(const Tensor3D& gammaGradients, const Tensor3D& betaGradients) {
        for (int i = 0; i < numLayers; ++i) {
            gamma.chip(i, 0) -= learningRate * gammaGradients.chip(1, 0);
            beta.chip(i, 0) -= learningRate * betaGradients.chip(1, 0);
        }
    }

protected:
    vector<SelfAttention> selfAttentions;
    vector<LinearProjection> linearProjections;

    Tensor3D gamma;
    Tensor3D beta;

    int numLayers;
    int numHeads;
    float learningRate;
    float clipNorm;

    float epsilon = 1e-5; 
};

// Encoder layer of the transformer network
class Encoder : public TransformerLayer {
public:
    Encoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed) {}

    Tensor3D feedForward(const Tensor3D& input) override {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {
            output = selfAttentions[i].feedForward(output);
            output = MathUtils::layerNormalisation(output, gamma, beta, epsilon);
            output = linearProjections[i].feedForward(output);
        }

        return output;
    }

    void updateParameters() override {}

};

// Decoder layer of the transformer network
class Decoder : public TransformerLayer {
public:
    Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed) {}

    Tensor3D feedForward(const Tensor3D& input) override {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {

            // masked multihead attention with add and norm

            output = selfAttentions[i].feedForward(output);
            output = MathUtils::layerNormalisation(output, gamma, beta, epsilon);
            output = linearProjections[i].feedForward(output);
        }
        
        return output;
    }

    void updateParameters() override {}

};