/* 
 * File:   MultilayerNN.cpp
 * Author: Austo89
 * 
 * Created on September 28, 2015, 7:18 AM
 */

#include "MultilayerNN.h"
#include <random>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string>

MultilayerNN::MultilayerNN(int _inputNodes, int _hiddenNodes, int _hiddenLayers, int _outputNodes,
                           string _activateHidden, string _activateOutput, double _momentum, double _learningRate) {
    hiddenLayerCount = _hiddenLayers;
    hiddenNodesPerLayer = _hiddenNodes;
    outputNodesN = _outputNodes;
    inputNodesN = _inputNodes;
    momentum = _momentum;
    learningRate = _learningRate;
    if (_activateHidden.compare("sigmoid") == 0) hiddenSigmoid = true;
    if (_activateOutput.compare("sigmoid") == 0) outputSigmoid = true;

    // Setup network structure
    setTopology();
}

MultilayerNN::MultilayerNN(const MultilayerNN &orig) {
}

MultilayerNN::~MultilayerNN() {
}

void MultilayerNN::setTopology() {
    random_device rd;                                               // Initialize random device & distribution
    uniform_real_distribution<double> dist(-0.3f, 0.3f);
    outputNodes.resize(outputNodesN);                               // We create N output node
    inputNodes.resize(inputNodesN);                            // One node per input
    hiddenNodes.resize(hiddenLayerCount);                           // Set hidden layers
    for (auto &layer : hiddenNodes) {
        layer.resize(hiddenNodesPerLayer);                          // Set hidden nodes in each layer
    }

    // Setup weights vector
    weights.resize(hiddenLayerCount + 1);                             // One set of weights between each layer
    if (hiddenLayerCount > 0) {
        weights.at(0).resize(inputNodes.size() * hiddenNodesPerLayer);    // Weights between input and hidden(0)
    } else {
        weights.at(0).resize(inputNodes.size() * outputNodes.size());
    }
    for (int l = 1; l <= hiddenLayerCount; l++) {                    // Weights between hidden layers
        weights.at(l).resize(hiddenNodesPerLayer * hiddenNodesPerLayer);
    }
    if (hiddenLayerCount > 0) {
        weights.back().resize(hiddenNodesPerLayer * outputNodes.size());   // Weights between hidden and output
    }

    // Randomize weights
    // Input layer to first hidden layer
    for (auto &weight : weights.at(0)) {
        weight = dist(rd);
    }

    // Between hidden layers
    for (int l = 1; l <= hiddenLayerCount; l++) {
        for (auto &weight : weights.at(l)) {
            weight = dist(rd);
        }
    }

    // Hidden to output
    for (auto &weight : weights.at(hiddenLayerCount)) {
        weight = dist(rd);
    }
}

void MultilayerNN::feedForward() {

    // If no hidden layers, feed from input to output layers
    if (hiddenLayerCount < 1) {
        // Loop through each output node
        for (int outNode = 0; outNode < outputNodes.size(); outNode++) {
            // Zero out output node
            outputNodes.at(outNode) = 0;
            // Loop through nodes in input layer
            for (int inNode = 0; inNode < inputNodes.size(); inNode++) {
                outputNodes.at(outNode) += inputNodes.at(inNode) * weights.at(0).at(inNode);
            }
            // Activate that shit
            if (outputSigmoid) {
                outputNodes.at(outNode) = activate(outputNodes.at(outNode));
            }
        }
    } else {
        // Feed through hidden layers
        // First we go from input layer to first hidden layer
        // Loop per hidden node
        for (int hiddenNode = 0; hiddenNode < hiddenNodesPerLayer; hiddenNode++) {
            // Zero out hidden node
            hiddenNodes.at(0).at(hiddenNode) = 0.0f;
            // Then per input node
            for (int inNode = 0; inNode < inputNodes.size(); inNode++) {
                hiddenNodes.at(0).at(hiddenNode) += inputNodes.at(inNode) * weight_input_hidden(inNode, hiddenNode);
            }
            // Activate hidden node!!!
            if (hiddenSigmoid) {
                hiddenNodes.at(0).at(hiddenNode) = activate(hiddenNodes.at(0).at(hiddenNode));
            }
        }

        // Next we feed between all hidden layers: hidLay represents hiddenNodes vector index
        // Runs only if more than one hidden layer
        for (int hidLay = 1; hidLay < hiddenLayerCount; hidLay++) {
            // For each node in hidLay, calculate S
            for (int thisNode = 0; thisNode < hiddenNodesPerLayer; thisNode++) {
                // Zero out node
                hiddenNodes.at(hidLay).at(thisNode) = 0;
                // Loop through each node in previous hidden layer to calc S
                for (int prevNode = 0; prevNode < hiddenNodesPerLayer; prevNode++) {
                    // Multiply value of node in this layer -1 by weight connecting these two layers
                    hiddenNodes.at(hidLay).at(thisNode) +=
                            hiddenNodes.at(hidLay - 1).at(prevNode) * weight_hidden_hidden(hidLay, prevNode, thisNode);
                }
                // Activate this node!!!
                if (hiddenSigmoid) {
                    hiddenNodes.at(hidLay).at(thisNode) = activate(hiddenNodes.at(hidLay).at(thisNode));
                }
            }
        }

        // Now we feed from last hidden layer to output
        // Loop through each output node
        for (int outNode = 0; outNode < outputNodes.size(); outNode++) {
            // Zero out output node
            outputNodes.at(outNode) = 0;
            // Loop through nodes in last hidden layer, find S for this output node
            for (int prevNode = 0; prevNode < hiddenNodesPerLayer; prevNode++) {
                outputNodes.at(outNode) += hiddenNodes.back().at(prevNode) * weight_hidden_output(prevNode, outNode);
            }
            // Activate that shit
            if (outputSigmoid) {
                outputNodes.at(outNode) = activate(outputNodes.at(outNode));
            }
        }
    }
}

double MultilayerNN::calculateOutputError(double target) {
    double squaredError = 0;

    // Synthesize target for each output node
    vector<double> nodeTargets(outputNodes.size());

    // If we have one output node (i.e. function)
    if (outputNodesN < 2) {
        nodeTargets.front() = target;

        // Otherwise we're classifying, so assign proper 0/1s
    } else {
        // Set all targets to 0
        fill(nodeTargets.begin(), nodeTargets.end(), 0);
        // Set target node to 1
        nodeTargets.at(target) = 1;
    }

    // Calculate error for output nodes
    for (int outNode = 0; outNode < outputNodes.size(); outNode++) {
        // Error = target value minus actual value
        squaredError += pow((nodeTargets.at(outNode) - outputNodes.at(outNode)), 2.0);
    }
    return squaredError / (2 * outputNodes.size());
}

double MultilayerNN::activate(double S) {
    return tanh(S);
}

double MultilayerNN::run(vector<vector<double>> data) {
    //ofstream dataWriter;
    //ofstream dataWriter2;

    //dataWriter.open("nnOutput.txt", ofstream::out | ofstream::trunc);
    //dataWriter2.open("weights.txt", ofstream::out | ofstream::trunc);

    double mse = 0;

    for (int i = 0; i < data.size(); i++) {
        mse += runOne(data.at(i));
    }

    mse /= data.size();

    return mse;
}

double MultilayerNN::runOne(vector<double> tuple) {
    // Set input nodes to testing tuple values
    for (int i = 0; i < inputNodes.size(); i++) {
        inputNodes.at(i) = tuple.at(i);
    }
    // Run tuple through net
    feedForward();
    // Return squared error
    return calculateOutputError(tuple.back());
}