#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>
#include "transformer.h"
#include "preprocessing.h"
#include "layers.h"

using namespace std;
using json = nlohmann::json;

// Global data/objects
json config;
Encoder* encoderPtr = nullptr;
Decoder* decoderPtr = nullptr;
TransformerModel* transformerPtr = nullptr;

// Function to parse the command and its arguments
void parseCommand(const string& input, string& command, vector<string>& args) {
    args.clear(); // Clear the arguments vector
    istringstream iss(input);
    iss >> command;
    string arg;
    while (iss >> arg) {
        args.push_back(arg);
    }
}

// Function to clear the screen
void clearScreen() {
    #ifdef _WIN32
        system("cls"); // For Windows
    #else
        system("clear"); // For Unix-based systems
    #endif
}

// Function to set the terminal title
void setTerminalTitle(const string& title) {
    #ifdef _WIN32
        system(("title " + title).c_str());
    #else
        cout << "\033]0;" << title << "\007";
    #endif
}

// Function to initialise a neural network instance
void intialiseNeuralNetwork() {
    if (config.empty()) {
        cout << "Error: Configuration data is missing. Please upload config data first." << endl;
    }

    try {
        // Encoder parameters
        encoderPtr = new Encoder(
        config["encoder"]["num_layers"],
        config["encoder"]["input_size"],
        config["encoder"]["output_size"],
        config["encoder"]["num_heads"],
        config["encoder"]["learning_rate"],
        config["encoder"]["clip_norm"],
        config["encoder"]["attention_scaling"],
        config["encoder"]["seed"]);
        cout << "Encoder initialisation complete." << endl;

        // Decoder parameters
        decoderPtr = new Decoder(
        config["decoder"]["num_layers"],
        config["decoder"]["input_size"],
        config["decoder"]["output_size"],
        config["decoder"]["num_heads"],
        config["decoder"]["learning_rate"],
        config["decoder"]["clip_norm"],
        config["decoder"]["attention_scaling"],
        config["decoder"]["masked_attention_scaling"],
        config["decoder"]["seed"]);
        cout << "Decoder initialisation complete." << endl;

        transformerPtr = new TransformerModel(*encoderPtr, *decoderPtr);
        cout << "Network initialisation complete." << endl;

    } catch(const nlohmann::json::exception& e) {
        cerr << "Error: " << e.what() << endl;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}

// Function to train the neural network
void trainNeuralNetwork(const string& filePath, int& epochs) {
    if (transformerPtr == nullptr) {
        cout << "Error: Neural network is not initialised. Please initialise the network first." << endl;
        return;
    }

    if (epochs <= 0) {
        cout << "Error: Invalid number of epochs. Please specify a positive integer greater than 0." << endl;
        return;
    } 

    try {
        vector<vector<string>> trainingData = DataIO::loadTrainingData(filePath);
        if (trainingData.size() != 0) {
            map<string, int> vocabulary = SequenceEncoding::generateVocabulary(trainingData[0]);
            vocabulary = SequenceEncoding::addToVocabulary(vocabulary, trainingData[1]);
            vector<Tensor3D> encodedContextSequences = SequenceEncoding::encodeSequences(trainingData[0], vocabulary, config["model"]["max_length"], config["model"]["embedding_size"]);
            vector<Tensor3D> encodedResponseSequences = SequenceEncoding::encodeSequences(trainingData[1], vocabulary, config["model"]["max_length"], config["model"]["embedding_size"]);
            transformerPtr->train(encodedContextSequences, encodedResponseSequences, epochs);
        }

    } catch(const nlohmann::json::exception& e) {
        cerr << "Error: " << e.what() << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}

// Function to display help information
void displayHelp() {
    cout << "Available commands:" << endl;
    cout << "  help                - Display this help message" << endl;
    cout << "  echo <message>      - Print the message to the console" << endl;
    cout << "  exit                - Exit the program" << endl;
    cout << "  cls, clear          - Clear the screen" << endl;
    cout << "  set_title <title>   - Set the terminal window title" << endl;
    cout << "  upload_config <file_path> - Upload configuration settings from a file" << endl;
    cout << "  show_config         - Display the current loaded configuration" << endl;
    cout << "  create_nn           - Initialize the neural network" << endl;
    cout << "  train_nn <file_path> [--epochs <num_epochs>] - Train the neural network" << endl;
}

// Function to display the loaded configuration
void displayConfig() {
    if (config.empty()) {
        cout << "No configuration loaded." << endl;
    } else {
        cout << "Current configuration:" << endl;
        cout << config.dump(4) << endl;
    }
}

// Function to execute the command
void executeCommand(const string& command, const vector<string>& args) {
    if (command == "help") {
        displayHelp();
    } else if (command == "echo") {
        for (const string& arg : args) {
            cout << arg << " ";
        }
        cout << endl;
    } else if (command == "exit") {
        cout << "Exiting..." << endl;
        exit(0);
    } else if (command == "cls" || command == "clear") {
        clearScreen();
    } else if (command == "set_title") {
        if (args.size() < 1) {
            cout << "Usage: set_title <title>" << endl;
        } else {
            setTerminalTitle(args[0]);
        }
    } else if (command == "create_nn") {
        intialiseNeuralNetwork();
    } else if (command == "upload_config") {
        if (args.size() < 1) {
            cout << "Usage: upload_config <file_path>" << endl;
        } else {
            config = DataIO::loadConfig(args[0]);
        }
    } else if (command == "show_config") {
        displayConfig();
    } else if (command == "train_nn") {
        if (args.size() < 1) {
            cout << "Usage: train_nn <file_path>" << endl;
        } else {
            int epochs = config["training"]["epochs"]; // Default value from config
            string filePath = args[0];
            try {
            trainNeuralNetwork(filePath, epochs);
            } catch (const exception& e) {
                cout << "Error: " << e.what() << endl;
            }
        }
    } else {
        cout << "Invalid command: " << command << endl;
    }
}

int main() {
    string userInput;
    string command;
    vector<string> args;

    setTerminalTitle("TranArc - Alpha Build v0.1.0");

    while (true) {
        cout << ">> ";
        getline(cin, userInput); // Read entire line including spaces
        parseCommand(userInput, command, args);
        executeCommand(command, args);
    }

    // Cleaning-up
    if (transformerPtr != nullptr) {
        delete transformerPtr;
        transformerPtr = nullptr;
    }

    if (encoderPtr != nullptr) {
        delete encoderPtr;
        encoderPtr = nullptr;
    }

    if (decoderPtr != nullptr) {
        delete decoderPtr;
        decoderPtr = nullptr;
    }

    return 0;
}
