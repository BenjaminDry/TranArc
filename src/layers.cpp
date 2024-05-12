#include <vector>
#include <Eigen/Dense>
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

    Tensor3D feedForward(const Tensor3D& input) {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {
            // Self attention
            output += selfAttentions[i].feedForward(output, Tensor3D());
            output = MathUtils::layerNormalisation(output, gamma, beta, epsilon);

            // Feed forward linear projection
            output += linearProjections[i].feedForward(output);
        }

        return output;
    }

    void updateParameters() override {}

};

// Decoder layer of the transformer network
class Decoder : public TransformerLayer {
public:
    Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed),
    outputProjection(outputSize, outputSize, learningRate, clipNorm, seed),
    mask(generateDecoderMask(inputSize)) {
        maskedSelfAttentions.reserve(numLayers);
        for (int i = 0; i < numLayers; ++i) {
            maskedSelfAttentions.push_back(SelfAttention(inputSize, numHeads));
        }
    }

    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& encoderOutput) {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {
            // Masked self attention
            output += maskedSelfAttentions[i].feedForward(output, mask);
            output = MathUtils::layerNormalisation(output, gamma, beta, epsilon);

            // Self attention
            Tensor3D combinedInput = MathUtils::concatenate(output, encoderOutput);
            output += selfAttentions[i].feedForward(combinedInput, Tensor3D());
            output = MathUtils::layerNormalisation(output, gamma, beta, epsilon);

            // Feed forward linear projection
            output += linearProjections[i].feedForward(output);
        }
        
        // Output linear projection and activation
        output = outputProjection.feedForward(output);
        output = MathUtils::softmax(output, 2);

        return output;
    }

    void updateParameters() override {}

private:
    vector<SelfAttention> maskedSelfAttentions;
    LinearProjection outputProjection;
    Tensor3D mask;

    // Compute the mask for the masked multiheader attention decoder layer
    Tensor3D generateDecoderMask(int inputSize) {
        Tensor3D mask(1, inputSize, inputSize);
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                if (j >= i) {
                    mask(0, i, j) = 0;  // Allow past and current positions
                } else {
                    mask(0, i, j) = -numeric_limits<float>::infinity();  // Remove future positions
                }
            }
        }
        return mask;
    }
};