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
        Tensor3D queries, keys, values;
        computeQKV(input, queries, keys, values);

        // Compute self-attention scores
        Tensor3D attentionScores = computeSelfAttention(queries, keys, values, mask);

        // Apply the attention scores to values
        Tensor3D weightedValues = attentionScores.contract(values, Eigen::array<IndexPair<long>, 2>{IndexPair<long>(2, 1)});

        // Sum all attention heads
        Tensor3D output = weightedValues.sum(Eigen::array<int, 1>{2});

        // Linear projection
        output = outputProjection.feedForward(output);
        return output;
    }

private:
    // Projection layers for dividing input into main data streams
    LinearProjection queryProjection;
    LinearProjection keyProjection;
    LinearProjection valueProjection;

    // Number of parallel attention mechanism
    int numHeads;

    // Final projection layer of the self attention mechanism
    LinearProjection outputProjection;

    // Compute the queries, keys and values of the self attention
    void computeQKV(const Tensor3D& input, Tensor3D& queries, Tensor3D& keys, Tensor3D& values) {
        queries = queryProjection.feedForward(input);
        keys = keyProjection.feedForward(input);
        values = valueProjection.feedForward(input);

        // Split tensors into multiple heads
        auto newShape = Eigen::array<Eigen::Index, 3>{
            queries.dimensions()[0],
            queries.dimensions()[1] / numHeads, numHeads
        };

        queries = queries.reshape(newShape);
        keys = keys.reshape(newShape);
        values = values.reshape(newShape);
    }

    // Comput the self attention of the input
    // This is not my code (smarter people created a smart solution)
    // The output tensor is computed by applying the self-attention mechanism
    Tensor3D computeSelfAttention(const Tensor3D& queries, const Tensor3D& keys, const Tensor3D& values, const Tensor3D& mask = Tensor3D()) {
        Tensor3D scores = queries.contract(keys, Eigen::array<IndexPair<long>, 1>{IndexPair<long>(1, 2)}).eval();
        if (mask.size() != 0) {scores += mask;}  // Masking
        scores = scores.unaryExpr([&scores](float x) { return MathUtils::softmax(x, 2); });
        return scores.contract(values, Eigen::array<IndexPair<long>, 2>{IndexPair<long>(2, 1)}).eval();
    }
};