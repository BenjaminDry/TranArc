#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>

using namespace std;
using namespace Eigen;

using Tensor3D = Tensor<float, 3>;

namespace MathUtils {
    VectorXd sigmoid(const VectorXd& x);
    Tensor3D softmax(const Tensor3D& x, int axis);
    void initialiseWeights(MatrixXd& weights, int& seed);
    float matrixClippingFactor(MatrixXd& gradient, float& clipNorm);
    MatrixXd reshapeToMatrix(const Tensor3D& tensor);
    Tensor3D reshapeToTensor(const MatrixXd& matrix, int dim1, int dim2, int dim3);
};

#endif
