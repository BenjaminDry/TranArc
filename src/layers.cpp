#include <vector>
#include <Eigen/Dense>
#include "attention.h"
#include "linear.h"
#include "math_utils.h"
#include "layers.h"

using namespace std;

// Base layer class constructor
TransformerLayer::TransformerLayer(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int scalingFactor, int seed)
    : numLayers(numLayers), numHeads(numHeads), learningRate(learningRate), clipNorm(clipNorm) {
    // Linear projection and self attention layer initialization
    for (int i = 0; i < numLayers; ++i) {
        selfAttentions.push_back(new SelfAttention(inputSize, numHeads, learningRate, clipNorm, scalingFactor, seed));
        linearProjections1.push_back(new LinearProjection(inputSize, outputSize, learningRate, clipNorm, seed));
        linearProjections2.push_back(new LinearProjection(outputSize, outputSize, learningRate, clipNorm, seed));
    }
}

// Base layer class destructor
TransformerLayer::~TransformerLayer() {
    // Clean up dynamically allocated memory
    for (auto& attention : selfAttentions) {
        delete attention;
    }
    for (auto& projection : linearProjections1) {
        delete projection;
    }
    for (auto& projection : linearProjections2) {
        delete projection;
    }
}

void TransformerLayer::initialiseLayerNormalisationValues(int numNormalisationLayers, int outputSize) {
    for (int i = 0; i < numNormalisationLayers; ++i) {
        MatrixXd gammaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.5);  // 0.5 by default
        MatrixXd betaMatrix = MatrixXd::Constant(numLayers * outputSize, 1, 0.1);   // 0.1 by default
        gamma.push_back(MathUtils::reshapeToTensor(gammaMatrix, numLayers, 1, outputSize));
        beta.push_back(MathUtils::reshapeToTensor(betaMatrix, numLayers, 1, outputSize));
    }
}

void TransformerLayer::updateNormalisationParameters(int parameterIndex, const Tensor3D& gammaGradients, const Tensor3D& betaGradients) {
    for (int i = 0; i < numLayers; ++i) {
        gamma[parameterIndex].chip(i, 0) -= learningRate * gammaGradients.chip(i, 0);
        beta[parameterIndex].chip(i, 0) -= learningRate * betaGradients.chip(i, 0);
    }
}

Encoder::Encoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int scalingFactor, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, scalingFactor, seed) {}

Encoder::~Encoder() {
    // No additional dynamic memory to clean up
}

Tensor3D Encoder::feedForward(const Tensor3D& input) {
    Tensor3D output = input;
    for (int i = 0; i < numLayers; ++i) {
        // Self attention
        output += selfAttentions[i]->feedForward(output, Tensor3D());
        output = MathUtils::layerNormalisation(output, gamma[0], beta[0], EPSILON);  // Current source of error (broken layer norm function)

        // Feed forward linear projection
        Tensor3D linearOutput = linearProjections1[i]->feedForward(output);
        output += linearProjections2[i]->feedForward(linearOutput);
        output = MathUtils::layerNormalisation(output, gamma[1], beta[1], EPSILON);
    }
    return output;
}

void Encoder::updateParameters(const Tensor3D& error) {
    Tensor3D normalisedError, gammaError, betaError;
    Tensor3D propagatedError = error;

    for (int i = numLayers - 1; i >= 0; --i) {
        // Normalization back propagation [2]
        Tensor3D linearOutput1 = linearProjections1[i]->getOutput();
        tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(linearOutput1, gamma[1], beta[1], propagatedError, EPSILON);
        updateNormalisationParameters(1, gammaError, betaError);

        // Linear projection back propagation [2]
        propagatedError = linearProjections2[i]->backPropagation(propagatedError);
        linearProjections2[i]->updateParameters(propagatedError);

        // Linear projection back propagation [1]
        propagatedError = linearProjections1[i]->backPropagation(propagatedError);
        linearProjections1[i]->updateParameters(propagatedError);
            
        // Normalization back propagation [1]
        Tensor3D attentionOutput = selfAttentions[i]->getLayerOutput();
        tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(attentionOutput, gamma[0], beta[0], propagatedError, EPSILON);
        updateNormalisationParameters(0, gammaError, betaError);
            
        // Multihead attention back propagation
        selfAttentions[i]->backPropagation(normalisedError);
    }
}

Decoder::Decoder(int numLayers, int inputSize, int outputSize, int numHeads, float learningRate, float clipNorm, int scalingFactor, int maskedScalingFactor, int seed)
    : TransformerLayer(numLayers, inputSize, outputSize, numHeads, learningRate, clipNorm, scalingFactor, seed),
      outputProjection(outputSize, outputSize, learningRate, clipNorm, seed),
      mask(generateDecoderMask(inputSize)) {
    // Masked self attention initialization
    maskedSelfAttentions.reserve(numLayers);
    for (int i = 0; i < numLayers; ++i) {
        maskedSelfAttentions.push_back(new SelfAttention(inputSize, numHeads, learningRate, clipNorm, maskedScalingFactor, seed));
    }
}

Decoder::~Decoder() {
    // Clean up dynamically allocated memory
    for (auto& attention : maskedSelfAttentions) {
        delete attention;
    }
}

Tensor3D Decoder::feedForward(const Tensor3D& input, const Tensor3D& newEncoderOutput) {
    Tensor3D output = input;
    encoderOutput = newEncoderOutput;

    for (int i = 0; i < numLayers; ++i) {
        // Masked self attention
        output += maskedSelfAttentions[i]->feedForward(output, mask);
        output = MathUtils::layerNormalisation(output, gamma[0], beta[0], EPSILON);

        // Self attention
        Tensor3D combinedInput = MathUtils::concatenate(output, encoderOutput);
        output += selfAttentions[i]->feedForward(combinedInput, Tensor3D());
        output = MathUtils::layerNormalisation(output, gamma[1], beta[1], EPSILON);

        // Feed forward linear projection
        Tensor3D linearOutput1 = linearProjections1[i]->feedForward(output);
        output += linearProjections2[i]->feedForward(linearOutput1);
        output = MathUtils::layerNormalisation(output, gamma[2], beta[2], EPSILON);
    }
    
    // Output linear projection and activation
    output = outputProjection.feedForward(output);
    output = MathUtils::softmax(output);

    return output;
}

void Decoder::updateParameters(const Tensor3D& error) {
    // Variables to store intermediate error/gradients
    Tensor3D normalisedError, gammaError, betaError;

    // Output projection back propagation
    Tensor3D propagatedError = outputProjection.backPropagation(error);
    outputProjection.updateParameters(propagatedError);

    for (int i = numLayers - 1; i >= 0; --i) {
        // Normalization back propagation [3]
        Tensor3D linearOutput2 = linearProjections2[i]->getOutput();
        tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(linearOutput2, gamma[2], beta[2], propagatedError, EPSILON);
        updateNormalisationParameters(2, gammaError, betaError);

        // Linear projection back propagation [2]
        propagatedError = linearProjections2[i]->backPropagation(propagatedError);
        linearProjections2[i]->updateParameters(propagatedError);

        // Linear projection back propagation [1]
        propagatedError = linearProjections1[i]->backPropagation(propagatedError);
        linearProjections1[i]->updateParameters(propagatedError);
            
        // Normalization back propagation [2]
        Tensor3D attentionOutput = selfAttentions[i]->getLayerOutput();
        tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(attentionOutput, gamma[1], beta[1], normalisedError, EPSILON);
        updateNormalisationParameters(1, gammaError, betaError);
            
        // Multihead attention back propagation
        selfAttentions[i]->backPropagation(normalisedError);

        // Set encoder error value
        encoderError = normalisedError;

        // Normalization back propagation [1]
        Tensor3D maskedAttentionOutput = maskedSelfAttentions[i]->getLayerOutput();
        tie(normalisedError, gammaError, betaError) = MathUtils::layerNormalisationBackPropagation(maskedAttentionOutput, gamma[0], beta[0], normalisedError, EPSILON);
        updateNormalisationParameters(0, gammaError, betaError);

        // Masked multihead attention back propagation
        maskedSelfAttentions[i]->backPropagation(normalisedError);
    }
    
    // Compute the error for the encoder
    encoderError.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({encoderOutput.dimension(1)}));
}

// Return the encoder error value from decoder back propagation
Tensor3D Decoder::getEncoderError() {
    return encoderError;
}

// Generate the mask for the masked multiheader attention decoder layer
Tensor3D Decoder::generateDecoderMask(int inputSize) {
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