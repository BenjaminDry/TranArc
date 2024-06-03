#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "transformer.h"
#include "layers.h"
#include "math_utils.h"

using namespace std;

using Tensor3D = Eigen::Tensor<float, 3>;

// Constructor definition
TransformerModel::TransformerModel(Encoder& encoderPtr, Decoder& decoderPtr)
: encoder(encoderPtr), decoder(decoderPtr) {}

// Destructor definition
TransformerModel::~TransformerModel() {}

// Function for training the neural network
void TransformerModel::train(const vector<Tensor3D>& inputSeq, vector<Tensor3D>& targetSeq, int epochs) {
    for (int e = 0; e < epochs; ++e) {
        float totalLoss = 0.0;
        for (int t = 0; t < inputSeq.size(); ++t) {
            // Batch size x Sequence length x Embedding size
            Tensor3D input = inputSeq[t]; 
            Tensor3D target = targetSeq[t];

            Tensor3D encoderOutput = encoder.feedForward(input);
            Tensor3D decoderOutput = decoder.feedForward(target, encoderOutput);

            cout << "ff" << endl;

            float loss = MathUtils::categorialCrossEntropyLoss(decoderOutput, target);
            totalLoss += loss;

            cout << "loss" << endl;

            Tensor3D decoderError = decoderOutput - target;
            decoder.updateParameters(decoderError);
            Tensor3D encoderError = decoder.getEncoderError();
            encoder.updateParameters(encoderError);
        }
        float averageLoss = totalLoss / inputSeq.size();
        cout << "Epoch: " << e + 1 << ", Avg Loss: " << averageLoss << endl;
    }
}

// Placeholder implementation for predict function
Tensor3D TransformerModel::predict(const Tensor3D& input) {
    return Tensor3D();
}