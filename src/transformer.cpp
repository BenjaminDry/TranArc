#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "layers.h"
#include "math_utils.h"

using namespace std;

using Tensor3D = Tensor<float, 3>;

class TransformerModel {
public:
    // Transformer model constructor
    TransformerModel(int numLayersEncoder, int inputSizeEncoder, int outputSizeEncoder, int numHeadsEncoder, float learningRateEncoder, float clipNormEncoder, int seedEncoder,
                     int numLayersDecoder, int inputSizeDecoder, int outputSizeDecoder, int numHeadsDecoder, float learningRateDecoder, float clipNormDecoder, int seedDecoder)
    : encoder(numLayersEncoder, inputSizeEncoder, outputSizeEncoder, numHeadsEncoder, learningRateEncoder, clipNormEncoder, seedEncoder),
      decoder(numLayersDecoder, inputSizeDecoder, outputSizeDecoder, numHeadsDecoder, learningRateDecoder, clipNormDecoder, seedDecoder) {}

    // Train the entire network architecture
    void train(const vector<Tensor3D>& inputSeq, vector<Tensor3D>& targetSeq, int epochs) {
        for (int e = 0; e < epochs; ++e) {
            float totalLoss = 0.0;
            for (int t = 0; t < inputSeq.size(); ++t) {
                Tensor3D input = inputSeq[t];
                Tensor3D target = targetSeq[t];

                Tensor3D encoderOutput = encoder.feedForward(input);
                Tensor3D decoderOutput = decoder.feedForward(target, encoderOutput);

                float loss = MathUtils::categorialCrossEntropyLoss(decoderOutput, target);
                totalLoss += loss;

                Tensor3D decoderError = decoderOutput - target;
                decoder.updateParameters(decoderError);
                Tensor3D encoderError;
                encoder.updateParameters(encoderError);
            }
            float averageLoss = totalLoss / inputSeq.size();
            cout << "Epoch: " << e + 1 << ", Avg Loss: " << averageLoss << endl;
        }
    }

    // Compute a predicted output based on the input prompt
    Tensor3D predict(const Tensor3D& input) {
        // ...
    }

private:
    // Encoder and Decoder instances
    Encoder encoder;
    Decoder decoder;
};