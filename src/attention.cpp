#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "linear.h"

using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

class SelfAttention {
public:
    SelfAttention(int inputSize, int numHeads, float learningRate, float clipNorm, int seed)
    : queryProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      keyProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      valueProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      outputProjection(inputSize, inputSize, learningRate, clipNorm, seed),
      numHeads(numHeads) {}

    // Calculate output of the self attention layer
    Tensor3D feedForward(const Tensor3D& input, const Tensor3D& mask = Tensor3D()) {
        layerInput = input;
        computeQKV();

        // Compute self-attention scores
        Tensor3D attentionScores = computeSelfAttention(mask);

        // Apply the attention scores to values
        Tensor3D weightedValues = attentionScores.contract(values, Eigen::array<IndexPair<long>, 2>{IndexPair<long>(2, 1)});

        // Concatenate all attention heads
        Tensor3D mergedValues = MathUtils::concatenate(weightedValues, values);

        // Linear projection
        layerOutput = outputProjection.feedForward(mergedValues);
        return layerOutput;
    }

    // Compute error of each section of the self attention layer and update parameters
    void backPropagation(const Tensor3D& prevError) {
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

    // Bandage fix for layer normalisation back propagation
    Tensor3D getLayerOutput() {
        return layerOutput;
    }

private:
    // Projection layers for dividing input into main data streams
    LinearProjection queryProjection;
    LinearProjection keyProjection;
    LinearProjection valueProjection;

    // Storing QKV's
    Tensor3D queries;
    Tensor3D keys;
    Tensor3D values;
    Tensor3D mergedValues;

    // Number of parallel attention mechanism
    int numHeads;

    // Final projection layer of the self attention mechanism
    LinearProjection outputProjection;

    // IO Data storage
    Tensor3D layerInput;
    Tensor3D layerOutput;

    // Compute the queries, keys and values of the self attention
    void computeQKV() {
        queries = queryProjection.feedForward(layerInput);
        keys = keyProjection.feedForward(layerInput);
        values = valueProjection.feedForward(layerInput);

        // Split tensors into multiple heads
        auto newShape = Eigen::array<Eigen::Index, 3>{
            queries.dimensions()[0],
            queries.dimensions()[1] / numHeads, numHeads
        };

        queries = queries.reshape(newShape);
        keys = keys.reshape(newShape);
        values = values.reshape(newShape);
    }

    // Compute the self attention of the input
    // This is not my code (smarter people created a smart solution)
    // The output tensor is computed by applying the self-attention mechanism
    Tensor3D computeSelfAttention(const Tensor3D& mask = Tensor3D()) {
        Tensor3D scores = queries.contract(keys, Eigen::array<IndexPair<long>, 1>{IndexPair<long>(1, 2)}).eval();
        if (mask.size() != 0) { scores += mask; }  // Masking
        scores = MathUtils::softmax(scores, 2);
        return scores.contract(values, Eigen::array<IndexPair<long>, 2>{IndexPair<long>(2, 1)}).eval();
    }
};