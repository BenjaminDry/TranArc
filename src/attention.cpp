#include "attention.h"
#include "math_utils.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

SelfAttention::SelfAttention(int inputSize, int numHeads, float learningRate, float clipNorm, int scalingFactor, int seed)
    : queryProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      keyProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      valueProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      outputProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      numHeads(numHeads) {}

Tensor3D SelfAttention::feedForward(const Tensor3D& input, const Tensor3D& mask) {
    layerInput = input;
    computeQKV();

    // Compute self-attention scores
    Tensor3D attentionScores = computeSelfAttention(mask);  // IMPORTANT: Logic error that causes a NaN tensor

    // Linear projection
    layerOutput = outputProjection.feedForward(attentionScores);
    return layerOutput;
}

void SelfAttention::backPropagation(const Tensor3D& prevError) {
    Tensor3D outputGradient = outputProjection.backPropagation(prevError);
    outputProjection.updateParameters(prevError);

    // Backpropagate through concatenated head
    Tensor3D mergedValuesGradient = MathUtils::concatenate(prevError, outputGradient);

    // Backpropagate through softmax function
    Tensor3D softmaxAttentionScoresGradient(prevError.dimensions());
    for (int i = 0; i < prevError.dimension(0); ++i) {
        for (int j = 0; j < prevError.dimension(1); ++j) {
            Tensor3D slice = prevError.chip(i, 0).chip(j, 0).reshape(Eigen::array<Eigen::Index, 2>{1, prevError.dimension(2)});
            Tensor3D softmaxSlice = slice * (1.0 - slice);
            softmaxAttentionScoresGradient.chip(i, 0).chip(j, 0) = softmaxSlice.reshape(Eigen::array<Eigen::Index, 3>{1, 1, prevError.dimension(2)});
        }
    }

    // Backpropagate through the self-attention scores
    Tensor3D valuesGradient = softmaxAttentionScoresGradient.contract(keys, Eigen::array<IndexPair<long>, 1>{IndexPair<long>(1, 2)});
    Tensor3D keysGradient = softmaxAttentionScoresGradient.contract(values, Eigen::array<IndexPair<long>, 1>{IndexPair<long>(2, 1)});
    Tensor3D queriesGradient = keysGradient.contract(keys, Eigen::array<IndexPair<long>, 1>{IndexPair<long>(2, 1)});

    // Update parameters (keeping it in the same function for simplicity)
    valueProjection.updateParameters(valuesGradient);
    keyProjection.updateParameters(keysGradient);
    queryProjection.updateParameters(queriesGradient);
}

Tensor3D SelfAttention::getLayerOutput() {
    return layerOutput;
}

void SelfAttention::computeQKV() {
    // Linear projection layer
    queries = queryProjection.feedForward(layerInput);
    keys = keyProjection.feedForward(layerInput);
    values = valueProjection.feedForward(layerInput);

    splitQueries.clear();
    splitKeys.clear();
    splitValues.clear();

    int totalSize = queries.dimension(2);  // Consider the last dimension
    int headSize = totalSize / numHeads;

    // Split into heads
    for (int i = 0; i < numHeads; ++i) {
        Eigen::array<Eigen::Index, 3> startIndices = {0, 0, i * headSize};  // Adjust the start indices
        Eigen::array<Eigen::Index, 3> sizes = {queries.dimension(0), queries.dimension(1), headSize};

        Tensor3D slicedQueries = queries.slice(startIndices, sizes);
        Tensor3D slicedKeys = keys.slice(startIndices, sizes);
        Tensor3D slicedValues = values.slice(startIndices, sizes);

        splitQueries.push_back(slicedQueries);
        splitKeys.push_back(slicedKeys);
        splitValues.push_back(slicedValues);
    }
}

Tensor3D SelfAttention::computeSelfAttention(const Tensor3D& mask) {
    int headSize = queries.dimension(1) / numHeads; 
    Tensor3D concatenatedAttention(layerInput.dimension(0), layerInput.dimension(1), queries.dimension(2));

    for (int i = 0; i < numHeads; ++i) {
        // TODO: Make a universial variable type! double or float!
        // Convert tensors to matrices
        MatrixXf queriesMatrix = MathUtils::reshapeToMatrix(splitQueries[i]).cast<float>();
        MatrixXf keysMatrix = MathUtils::reshapeToMatrix(splitKeys[i]).cast<float>();
        MatrixXf valuesMatrix = MathUtils::reshapeToMatrix(splitValues[i]).cast<float>();

        // Compute attention scores using matrix multiplication
        MatrixXf scoresMatrix = (queriesMatrix.transpose() * keysMatrix);  // Redo back propagation?
        scoresMatrix /= (1.0 * sqrt(headSize) * scalingFactor);
        
        // Apply mask (if provided), this might be cause of future error
        if (mask.size() != 0) {
            cout << "masking" << endl;
            MatrixXf maskMatrix = MathUtils::reshapeToMatrix(mask).cast<float>();  // I hate casting to float
            scoresMatrix += maskMatrix;
        }

        // Matrix based softmax activation
        MatrixXf scoresMatrixExp = scoresMatrix.array().exp();
        VectorXf sumScoresExp = scoresMatrixExp.rowwise().sum();
        scoresMatrix = scoresMatrixExp.array().colwise() / sumScoresExp.array();

        // Compute final attention output using matrix multiplication
        MatrixXf attentionMatrix = scoresMatrix * valuesMatrix.transpose();

        // Copy the attention output back to the tensor
        TensorMap<Tensor3D> attentionTensor(attentionMatrix.data(), 1, headSize, attentionMatrix.cols());
        concatenatedAttention.slice(Eigen::array<Index, 3>({0, i * headSize, 0}), Eigen::array<Index, 3>({1, headSize, attentionMatrix.cols()})) = attentionTensor;
    }
    return concatenatedAttention;
}