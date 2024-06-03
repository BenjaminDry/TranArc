#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlohmann/json.hpp>
#include "preprocessing.h"

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

using Tensor3D = Tensor<float, 3>;

// Functions for encoding sequences
namespace SequenceEncoding {

    // Function to add positional encoding to input embeddings
    Tensor3D addPositionalEncoding(const Tensor3D& input, int maxSequenceLength, int embeddingSize) {
        Tensor3D encodedInput = input;
        for (int pos = 0; pos < maxSequenceLength; ++pos) {
            for (int i = 0; i < embeddingSize; ++i) {
                float angle = pos / pow(10000, 2.0 * (i / 2) / embeddingSize);
                if (i % 2 == 0) {
                    encodedInput(0, pos, i) += sin(angle);
                } else {
                    encodedInput(0, pos, i) += cos(angle);
                }
            }
        }
        return encodedInput;
    }

    // Function to generate the vocabulary map
    map<string, int> generateVocabulary(const vector<string>& data) {
        map<string, int> vocabulary;
        int index = 0;

        // Add special tokens to the vocabulary
        vector<string> specialTokens = {"<SOS>", "<EOS>", "<UNK>", "<PAD>"};
        for (const string& token : specialTokens) {
            vocabulary[token] = index++;
        }

        // Add tokens from training data to the vocabulary
        for (const string& sequence : data) {
            vector<string> tokens = tokenise(sequence);
            for (const string& token : tokens) {
                if (vocabulary.find(token) == vocabulary.end()) {
                    vocabulary[token] = index++;
                }
            }
        }
        return vocabulary;
    }

    // Function to add tokens to the vocabulary map
    map<string, int> addToVocabulary(map<string, int>& vocabulary, const vector<string>& data) {
        int index = vocabulary.size();
        for (const string& sequence : data) {
            vector<string> tokens = tokenise(sequence);
            for (const string& token : tokens) {
                if (vocabulary.find(token) == vocabulary.end()) {
                    vocabulary[token] = index++;
                }
            }
        }
        return vocabulary;
    }

    // Function to tokenise text into words
    vector<string> tokenise(const string& text) {
        vector<string> tokens;
        istringstream iss(text);
        string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Function to encode tokens using vocabulary
    Tensor3D encodeTokens(const vector<string>& tokens, const map<string, int>& vocabulary, int embeddingSize) {
        int tokenCount = tokens.size();
        Tensor3D encodedSequence(1, tokenCount, embeddingSize);
        encodedSequence.setZero(); // Initialize the tensor with zeros

        // Encode each token
        for (int i = 0; i < tokenCount; ++i) {
            const string& token = tokens[i];
            int index = (vocabulary.find(token) != vocabulary.end()) ? vocabulary.at(token) : vocabulary.at("<UNK>");

            // Fill the slice with the index
            for (int j = 0; j < embeddingSize; ++j) {
                encodedSequence(0, i, j) = index;
            }
        }
        return encodedSequence;
    }

    // Function to encode sequences as tensors
    vector<Tensor3D> encodeSequences(const vector<string>& sequences, const map<string, int>& vocabulary, int maxSequenceLength, int embeddingSize) {
        vector<Tensor3D> encodedSequences;
        for (const string& sequence : sequences) {
            vector<string> tokens = tokenise(sequence);
            Tensor3D encodedSequence = encodeTokens(tokens, vocabulary, embeddingSize);

            // Pad sequences
            Tensor3D paddedSequence(1, maxSequenceLength, embeddingSize);
            paddedSequence.setZero();
            int length = min((int)tokens.size(), maxSequenceLength);
            paddedSequence.slice(Eigen::array<Index, 3>({0, 0, 0}), Eigen::array<Index, 3>({1, length, embeddingSize})) = encodedSequence.slice(Eigen::array<Index, 3>({0, 0, 0}), Eigen::array<Index, 3>({1, length, embeddingSize}));

            paddedSequence = addPositionalEncoding(paddedSequence, maxSequenceLength, embeddingSize);
            encodedSequences.push_back(paddedSequence);
        }
        return encodedSequences;
    }
}

// Functions for input/output operations
namespace DataIO {

    // Load config file data
    json loadConfig(const string& filePath) {
        ifstream file(filePath);
        json config;
        if (file.is_open()) {
            file >> config;
            file.close();
            cout << "Configuration loaded successfully from " << filePath << endl;
        } else {
            cerr << "Error: Could not open configuration file at " << filePath << endl;
        }
        return config;
    }

    // Load training data, vector of response : context pairs
    vector<vector<string>> loadTrainingData(const string& filePath) {
        ifstream file(filePath);
        json trainingData;

        if (file.is_open()) {
            file >> trainingData;
            file.close();
            cout << "Training data loaded successfully from " << filePath << endl;
        } else {
            cerr << "Error: Could not open training data file at " << filePath << endl;
            return {};
        }

        // Parse JSON into vector<vector<string>>
        vector<vector<string>> parsedData;
        for (const auto& entry : trainingData) {
            vector<string> conversation;
            conversation.push_back(entry["response"]);
            for (const auto& context : entry["context"]) {
                conversation.push_back(context);
            }
            parsedData.push_back(conversation);
        }
        return parsedData;
    }

    // Parameter importing/exporting
    void loadParameters() {}
    void saveParameters() {}
}