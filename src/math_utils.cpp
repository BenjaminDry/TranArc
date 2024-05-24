#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>

using namespace std;
using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

namespace MathUtils {
    // Compute softmax activation of 3D tensor
    Tensor3D softmax(const Tensor3D& x, int axis) {
        Tensor3D xExp = (-x).exp();
        Tensor3D sumExp = xExp.sum(axis);
        Eigen::array<Eigen::Index, 3> reshapedDims = {1, 1, x.dimension(axis)};
        sumExp = sumExp.reshape(reshapedDims);
        return xExp / sumExp.broadcast(xExp.dimensions());
    }

    // Compute weight matrix, using random seed
    void initialiseWeights(MatrixXd& weights, int& seed) {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 1.0);
            
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.cols(); ++j) {
                weights(i, j) = distribution(generator);
            }
        }
    }

    // Compute scale factor for gradient clipping
    float matrixClippingFactor(const MatrixXd& gradients, const float& clipNorm) {
        float maxNorm = gradients.norm();
        if (maxNorm > clipNorm) {
            return clipNorm / maxNorm;
        }
        return 1.0;
    }

    // Reshape a 3D tensor to 2D matrix
    MatrixXd reshapeToMatrix(const Tensor3D& tensor) {
        int rows = tensor.dimension(0) * tensor.dimension(1);
        int cols = tensor.dimension(2);
        MatrixXd result(rows, cols);
        int index = 0;
        for (int i = 0; i < tensor.dimension(0); ++i) {
            for (int j = 0; j < tensor.dimension(1); ++j) {
                for (int k = 0; k < tensor.dimension(2); ++k) {
                    result(index, k) = tensor(i, j, k);
                }
                index++;
            }
        }
        return result;
    }

    // Reshape a 2D matrix to a 3D tensor
    Tensor3D reshapeToTensor(const MatrixXd& matrix, int dim1, int dim2, int dim3) {
        Tensor3D result(dim1, dim2, dim3);
        int index = 0;
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                for (int k = 0; k < dim3; ++k) {
                    result(i, j, k) = matrix(index, k);
                }
                index++;
            }
        }
        return result;
    }

    // Normalise layer data
    // LayerNorm(X) = gamma * (X - mean) / standard deviation + beta
    // Where gamma and beta are learnable vector parameters
    Tensor3D layerNormalisation(const Tensor3D& input, const Tensor3D& gamma, const Tensor3D& beta, float epsilon) {
        Eigen::array<int, 1> featureDimension = {2};  // 2 Corresponds to feature dimension (e.g. vocabulary index)
        Eigen::array<Eigen::Index, 1> dimensionSizeArray = {1};

        Tensor3D mean = input.mean(featureDimension).reshape(dimensionSizeArray);
        Tensor3D variance = ((input - mean.broadcast(input.dimensions())).square().mean(featureDimension)).reshape(dimensionSizeArray);

        Tensor3D normalised = (input - mean.broadcast(input.dimensions())) / (variance.broadcast(input.dimensions()) + epsilon).sqrt();;

        Tensor3D scaled = normalised * gamma.broadcast(input.dimensions());
        Tensor3D shifted = scaled + beta.broadcast(input.dimensions());

        return scaled;
    }

    // Backpropagation of the layer normalisation method, might move out of the math utils
    tuple<Tensor3D, Tensor3D, Tensor3D> layerNormalisationBackPropagation(const Tensor3D& input, const Tensor3D& gamma, const Tensor3D& beta, const Tensor3D& prevError, float epsilon) {
        Eigen::array<int, 1> featureDimension = {2};  // 2 Corresponds to feature dimension (e.g. vocabulary index)

        Tensor3D inverseStandardDeviation = (input.square().mean(featureDimension) + epsilon).sqrt().inverse();
        Tensor3D shiftedGradient = prevError;
        Tensor3D scaledGradient = shiftedGradient * gamma.broadcast(input.dimensions());
        Tensor3D normalisedGradient = scaledGradient * inverseStandardDeviation.broadcast(input.dimensions());

        Tensor3D gammaGradient = (prevError * input).sum(featureDimension);
        Tensor3D betaGradient = prevError.sum(featureDimension);

        return make_tuple(normalisedGradient, gammaGradient, betaGradient);
    }

    // Function to update the normalization parameters for a given layer
    void updateNormalisationParameters(Tensor3D& gamma, Tensor3D& beta, int layerIndex, const Tensor3D& gammaGradients, const Tensor3D& betaGradients, float learningRate) {
        gamma.chip(layerIndex, 0) -= learningRate * gammaGradients.chip(layerIndex, 0);
        beta.chip(layerIndex, 0) -= learningRate * betaGradients.chip(layerIndex, 0);
    }

    // Function to merge a tensor object on the end of another tensor object
    Tensor3D concatenate(const Tensor3D& input1, const Tensor3D& input2) {
        std::array<int, 3> newDimensions = {
            static_cast<int>(input1.dimension(0)),
            static_cast<int>(input1.dimension(1)),
            static_cast<int>(input1.dimension(2) + input2.dimension(2))
        };
        Tensor3D concatenated(newDimensions[0], newDimensions[1], newDimensions[2]);

        // Copy data from input1
        for (int i = 0; i < input1.dimension(0); ++i) {
            for (int j = 0; j < input1.dimension(1); ++j) {
                for (int k = 0; k < input1.dimension(2); ++k) {
                    concatenated(i, j, k) = input1(i, j, k);
                }
            }
        }

        // Copy data from input2
        for (int i = 0; i < input2.dimension(0); ++i) {
            for (int j = 0; j < input2.dimension(1); ++j) {
                for (int k = 0; k < input2.dimension(2); ++k) {
                    concatenated(i, j, input1.dimension(2) + k) = input2(i, j, k);
                }
            }
        }

        return concatenated;
    }

    // Compute categorial cross-entropy loss    
    float categorialCrossEntropyLoss(const Tensor3D& prediction, const Tensor3D& target) {
        float totalLoss = 0.0;

        // Iterate over every element in the batch
        for (int i = 0; i < prediction.dimension(0); ++i) {  // Batch dimension
            for (int j = 0; j < prediction.dimension(1); ++j) {  // Sequence length dimension
                for (int k = 0; k < prediction.dimension(2); ++k) {  // Vocabulary dimension
                    float probability = prediction(i, j, k);
                    int targetIndex = static_cast<int>(target(i, j, 0));
                    totalLoss -= log(probability) * (k = targetIndex ? 1.0 : 0.0);
                }
            }
        }

        // Normalise loss
        int numElements = prediction.size();
        totalLoss /= numElements;

        return totalLoss;
    }

};   