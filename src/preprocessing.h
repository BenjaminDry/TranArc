#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

using Tensor3D = Eigen::Tensor<float, 3>;

namespace SequenceEncoding {
    Tensor3D addPositionalEncoding(const Tensor3D& input, int maxSequenceLength, int embeddingSize);
    map<string, int> generateVocabulary(const vector<string>& data);
    map<string, int> addToVocabulary(map<string, int>& vocabulary, const vector<string>& data);
    vector<string> tokenise(const string& text);
    Tensor3D encodeTokens(const vector<string>& tokens, const map<string, int>& vocabulary, int embeddingSize);
    vector<Tensor3D> encodeSequences(const vector<string>& sequences, const map<string, int>& vocabulary, int maxSequenceLength, int embeddingSize);
}

namespace DataIO {
    json loadConfig(const string& filePath);
    vector<vector<string>> loadTrainingData(const string& filePath);
    void loadParameters();
    void saveParameters();
}

#endif