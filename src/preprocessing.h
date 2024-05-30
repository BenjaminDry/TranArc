#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

using Tensor3D = Tensor<float, 3>;

namespace SequenceEncoding {
    Tensor3D addPositonalEncoding(const Tensor3D& input, int maxSequenceLength, int embeddingSize);
}

namespace DataIO {
    json loadConfig(const string& filePath);
    void loadTrainingData();
    void loadParameters();
    void saveParameters();
}

#endif