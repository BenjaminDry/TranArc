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
        // Linear projection and self attention layer initalisation
        for (int i = 0; i < numLayers; ++i) {
            selfAttentions.push_back(SelfAttention(inputSize, numHeads, learningRate, clipNorm, seed));
            linearProjections.push_back(LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
        }

        // Gamma intialisation (0.5 inital constant)
        MatrixXd gammaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.5);
        gamma = MathUtils::reshapeToTensor(gammaMatrix, numLayers, 1, outputSize);

        // Beta intialisation (0.1 inital constant)
        MatrixXd betaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.1);
        beta = MathUtils::reshapeToTensor(betaMatrix, numLayers, 1, outputSize);
    }

    // Update the parameters of the layers
    virtual void updateParameters(const Tensor3D& error) = 0;

    // Update the gamma and beta tensor objects
    virtual void updateNormalisationParameters(const Tensor3D& gammaGradients, const Tensor3D& betaGradients) {
        for (int i = 0; i < numLayers; ++i) {
            gamma.chip(i, 0) -= learningRate * gammaGradients.chip(1, 0);
            beta.chip(i, 0) -= learningRate * betaGradients.chip(1, 0);
        }
    }

    // Update the gamma and beta tensor objects for a specific layer
    virtual void updateNormalisationParameters(int layerIndex, const Tensor3D& gammaGradients, const Tensor3D& betaGradients) {
        gamma.chip(layerIndex, 0) -= learningRate * gammaGradients.chip(layerIndex, 0);
        beta.chip(layerIndex, 0) -= learningRate * betaGradients.chip(layerIndex, 0);
    }

protected:
    // Internal projection and multihead attention layers
    vector<SelfAttention> selfAttentions;
    vector<LinearProjection> linearProjections;

    // Normalisation learnable parameters
    // Add a second set of these parameters to decoder as it goes through two layer normalisation steps?
    Tensor3D gamma;
    Tensor3D beta;

    // Training config
    int numLayers;
    int numHeads;
    float learningRate;
    float clipNorm;

    // Constant
    const float epsilon = 1e-5; 
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

    // Back propagation and subsequent updating of all encoder parameters
    void updateParameters(const Tensor3D& error) override {
        for (int i = numLayers - 1; i >= 0; --i) {
            // Linear projection back propagation
            Tensor3D linearError = linearProjections[i].backPropagation(error);
            linearProjections[i].updateParameters(linearError);
            
            // Normalisation back propagation
            Tensor3D attentionOutput = selfAttentions[i].getLayerOutput();
            auto [normalisedError, gammaError, betaError] = MathUtils::layerNormalisationBackPropagation(attentionOutput, gamma, beta, linearError, epsilon);
            updateNormalisationParameters(i, gammaError, betaError);

            // Multihead attention back propagation
            selfAttentions[i].backPropagation(linearError);
        }
    }
};

// Decoder layer of the transformer network
class Decoder : public TransformerLayer {
public:
    Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, seed),
    outputProjection(outputSize, outputSize, learningRate, clipNorm, seed),
    mask(generateDecoderMask(inputSize)) {
        // Masked self attention initialisation
        maskedSelfAttentions.reserve(numLayers);
        for (int i = 0; i < numLayers; ++i) {
            maskedSelfAttentions.push_back(SelfAttention(inputSize, numHeads, learningRate, clipNorm, seed));
        }

        // Gamma2 intialisation (0.5 inital constant)
        MatrixXd gammaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.5);
        gamma2 = MathUtils::reshapeToTensor(gammaMatrix, numLayers, 1, outputSize);

        // Beta2 intialisation (0.1 inital constant)
        MatrixXd betaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.1);
        beta2 = MathUtils::reshapeToTensor(betaMatrix, numLayers, 1, outputSize);
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

    // Back propagation and subsequent updating of all decoder parameters
    void updateParameters(const Tensor3D& error) override {
        // Initialize error tensors for normalisation
        Tensor3D normalisedError1, normalisedError2;
        Tensor3D gammaError1, gammaError2;
        Tensor3D betaError1, betaError2;

        // Output projection back propagation
        Tensor3D outputLinearError = outputProjection.backPropagation(error);
        outputProjection.updateParameters(outputLinearError);

        for (int i = numLayers - 1; i >= 0; --i) {
            // Linear projection back propagation
            Tensor3D linearError = linearProjections[i].backPropagation(outputLinearError);
            linearProjections[i].updateParameters(linearError);

            // Normalisation back propagation [2]
            Tensor3D attentionOutput1 = selfAttentions[i].getLayerOutput();
            tie(normalisedError1, gammaError1, betaError1) = MathUtils::layerNormalisationBackPropagation(attentionOutput1, gamma, beta, error, epsilon);
            MathUtils::updateNormalisationParameters(gamma, beta, i, gammaError1, betaError1, learningRate);

            // Multihead attention back propagation
            selfAttentions[i].backPropagation(normalisedError1);

            // Normalisation back propagation [1]
            Tensor3D attentionOutput2 = selfAttentions[i].getLayerOutput();
            tie(normalisedError2, gammaError2, betaError2) = MathUtils::layerNormalisationBackPropagation(attentionOutput2, gamma2, beta2, normalisedError1, epsilon);
            MathUtils::updateNormalisationParameters(gamma2, beta2, i, gammaError2, betaError2, learningRate);

            // Masked multihead attention back propagation
            maskedSelfAttentions[i].backPropagation(normalisedError2);
        }
    }

private:
    // Decoder-specific layers
    vector<SelfAttention> maskedSelfAttentions;
    LinearProjection outputProjection;
    Tensor3D mask;

    // Decoder extra normalisation values
    Tensor3D gamma2;
    Tensor3D beta2;

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