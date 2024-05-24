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
            linearProjections1.push_back(LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
            linearProjections2.push_back(LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
        }
    }

    // Intialise the layer normalisation parameters
    void intialiseLayerNormalisationValues(int numNormalisationLayers, int outputSize) {
        for (int i = 0; i < numNormalisationLayers; ++i) {
            MatrixXd gammaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.5);  // 0.5 by default
            MatrixXd betaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.1);  // 0.1 by default
            gamma.push_back(MathUtils::reshapeToTensor(gammaMatrix, numLayers, 1, outputSize));
            beta.push_back(MathUtils::reshapeToTensor(betaMatrix, numLayers, 1, outputSize));
        }
    }

    // Update the parameters of the layers
    virtual void updateParameters(const Tensor3D& error) = 0;

    // Update the gamma and beta tensor objects
    virtual void updateNormalisationParameters(int parameterIndex, const Tensor3D& gammaGradients, const Tensor3D& betaGradients) {
        for (int i = 0; i < numLayers; ++i) {
            gamma[parameterIndex].chip(i, 0) -= learningRate * gammaGradients.chip(1, 0);
            beta[parameterIndex].chip(i, 0) -= learningRate * betaGradients.chip(1, 0);
        }
    }

protected:
    // Internal projection and multihead attention layers
    vector<SelfAttention> selfAttentions;
    vector<LinearProjection> linearProjections1;
    vector<LinearProjection> linearProjections2;

    // Normalisation learnable parameters
    // Add a second set of these parameters to decoder as it goes through two layer normalisation steps?
    vector<Tensor3D> gamma;
    vector<Tensor3D> beta;

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
            output = MathUtils::layerNormalisation(output, gamma[0], beta[0], epsilon);

            // Feed forward linear projection
            Tensor3D linearOutput = linearProjections1[i].feedForward(linearOutput);
            output += linearProjections2[i].feedForward(linearOutput);
            output = MathUtils::layerNormalisation(output, gamma[1], beta[1], epsilon);
        }

        return output;
    }

    // Back propagation and subsequent updating of all encoder parameters
    void updateParameters(const Tensor3D& error) override {
        // Variables to store intermediate error/gradients
        Tensor3D normalisedError, gammaError, betaError;
        Tensor3D propagatedError = error;

        for (int i = numLayers - 1; i >= 0; --i) {
            // Normalisation back propagation [2]
            Tensor3D linearOutput1 = linearProjections1[i].getOutput();
            tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(linearOutput1, gamma[1], beta[1], propagatedError, epsilon);
            updateNormalisationParameters(i, gammaError, betaError);

            // Linear projection back propagation [2]
            propagatedError = linearProjections2[i].backPropagation(normalisedError);
            linearProjections2[i].updateParameters(propagatedError);

            // Linear projection back propagation [1]
            propagatedError = linearProjections1[i].backPropagation(propagatedError);
            linearProjections1[i].updateParameters(propagatedError);
            
            // Normalisation back propagation [1]
            Tensor3D attentionOutput = selfAttentions[i].getLayerOutput();
            tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(attentionOutput, gamma[0], beta[0], propagatedError, epsilon);
            updateNormalisationParameters(i, gammaError, betaError);

            // Multihead attention back propagation
            selfAttentions[i].backPropagation(normalisedError);
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
    }

    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& encoderOutput) {
        Tensor3D output = input;

        for (int i = 0; i < numLayers; ++i) {
            // Masked self attention
            output += maskedSelfAttentions[i].feedForward(output, mask);
            output = MathUtils::layerNormalisation(output, gamma[0], beta[0], epsilon);

            // Self attention
            Tensor3D combinedInput = MathUtils::concatenate(output, encoderOutput);
            output += selfAttentions[i].feedForward(combinedInput, Tensor3D());
            output = MathUtils::layerNormalisation(output, gamma[1], beta[1], epsilon);

            // Feed forward linear projection
            Tensor3D linearOutput1 = linearProjections1[i].feedForward(output);
            output += linearProjections2[i].feedForward(linearOutput1);
            output = MathUtils::layerNormalisation(output, gamma[2], beta[2], epsilon);
        }
        
        // Output linear projection and activation
        output = outputProjection.feedForward(output);
        output = MathUtils::softmax(output, 2);

        return output;
    }

    // Back propagation and subsequent updating of all decoder parameters
    void updateParameters(const Tensor3D& error) override {
        // Variables to store intermediate error/gradients
        Tensor3D normalisedError, gammaError, betaError;
        Tensor3D propagatedError = error;

        // Output projection back propagation
        Tensor3D propagatedError = outputProjection.backPropagation(error);
        outputProjection.updateParameters(propagatedError);

        for (int i = numLayers - 1; i >= 0; --i) {
            // Normalisation back propagation [3]
            Tensor3D linearOutput2 = linearProjections2[i].getOutput();
            tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(linearOutput2, gamma[2], beta[2], propagatedError, epsilon);
            updateNormalisationParameters(2, gammaError, betaError);

            // Linear projection back propagation [2]
            propagatedError = linearProjections2[i].backPropagation(propagatedError);
            linearProjections2[i].updateParameters(propagatedError);

            // Linear projection back propagation [1]
            propagatedError = linearProjections1[i].backPropagation(propagatedError);
            linearProjections1[i].updateParameters(propagatedError);
                
            // Normalisation back propagation [2]
            Tensor3D attentionOutput = selfAttentions[i].getLayerOutput();
            tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(attentionOutput, gamma[1], beta[1], normalisedError, epsilon);
            updateNormalisationParameters(1, gammaError, betaError);
                
            // Multihead attention back propagation
            selfAttentions[i].backPropagation(normalisedError);

            // Normalisation back propagation [1]
            Tensor3D maskedAttentionOutput = maskedSelfAttentions[i].getLayerOutput();
            tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(maskedAttentionOutput, gamma[0], beta[0], normalisedError, epsilon);
            updateNormalisationParameters(0, gammaError, betaError);

            // Masked multihead attention back propagation
            maskedSelfAttentions[i].backPropagation(normalisedError);
        }
    }

private:
    // Decoder-specific layers
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