#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>

using namespace std;
using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

namespace MathUtils {
    // Compute sigmoid activation
    VectorXd sigmoid(const VectorXd& x) {
        return 1.0 / (1 + (-x).array().exp());
    };

    // Compute softmax activation of 3D tensor
    Tensor3D softmax(const Tensor3D& x, int axis) {
        Tensor3D expValues = (-x).exp();
        Tensor3D sumExp = expValues.sum(axis);
        Eigen::array<Eigen::Index, 3> reshapedDims = {1, 1, x.dimension(axis)};
        sumExp = sumExp.reshape(reshapedDims);
        return expValues / sumExp.broadcast(x.dimensions());
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
        Eigen::array<int, 1> featureDimension = {2};  // 2 Corresponds to feature dimension
        Eigen::array<Eigen::Index, 1> dimensionSizeArray = {1};

        Tensor3D mean = input.mean(featureDimension).reshape(dimensionSizeArray);
        Tensor3D variance = ((input - mean.broadcast(input.dimensions())).square().mean(featureDimension)).reshape(dimensionSizeArray);

        Tensor3D normalised = (input - mean.broadcast(input.dimensions())) / (variance.broadcast(input.dimensions()) + epsilon).sqrt();;

        Tensor3D scaled = normalised * gamma.broadcast(input.dimensions());
        Tensor3D shifted = scaled + beta.broadcast(input.dimensions());

        return scaled;
    }

};   