/* 
 * File:   MultilayerNN.h
 * Author: Austo89
 *
 * Created on September 28, 2015, 7:18 AM
 */

#include <vector>
#include <math.h>
#include <string>
#include "Updater.h"

using namespace std;

#ifndef MULTILAYERNN_H
#define	MULTILAYERNN_H

// Derivative of tanh
#define sech2(x)                                (1.0 - (tanh(x) * tanh(x)))

// Helper functions for finding a specific weight
// Input to first hidden
#define weight_input_hidden(input, hidden)      weights.at(0).at(hiddenNodesPerLayer * input + hidden)
#define past_weight_input_hidden(input, hidden) previousWeights.at(0).at(hiddenNodesPerLayer * input + hidden)
// Hidden to hidden
#define weight_hidden_hidden(layer, i, j)       weights.at(layer).at(i * hiddenNodesPerLayer + j)
#define past_weight_hidden_hidden(layer, i, j)  previousWeights.at(layer).at(i * hiddenNodesPerLayer + j)

// Hidden to output
#define weight_hidden_output(hidden, output)    weights.back().at(output * hiddenNodesPerLayer + hidden)
#define past_weight_hidden_output(hidden, output) previousWeights.back().at(output * hiddenNodesPerLayer + hidden)

// Helper functions to calculate deltas
// For output nodes
#define delta_output(outNode, target)           (target - outputNodesN.at(outNode));
// outputNodesN.at(outNode) * (1.0f- outputNodesN.at(outNode)) *


class MultilayerNN {

private:

    bool topoSet  = false;

    int inputNodesN;
    int outputNodesN;
    int hiddenLayerCount;
    int hiddenNodesPerLayer;
    double momentum;
    double learningRate;
    double targetMSE;
    bool hiddenSigmoid = false;
    bool outputSigmoid = false;
    int noDecreaseCount = 0;
    double lastError;
    vector<vector<double>> weights;          // Weight between layer i and j
    vector<vector<double>> tempWeights;
    vector<vector<double>> previousWeights;  // Stores weights from last pattern
    vector<double> inputNodes;               // Value of node in input layer
    vector<double> outputNodes;              // Value of nodes in output layer
    vector<vector<double>> hiddenNodes;      // Values of node x in layer i (hidden layers)
    vector<double> inputDelats;              // Deltas for input layer
    vector<double> outputDeltas;             // Deltas for output layer
    vector<vector<double>> hiddenDeltas;     // Deltas for hidden layers

    void setTopology();      // Creates network structure and randomizes weights
    void feedForward();                     // Calculate a network given training tuple
    double calculateOutputError(double target);// Calculates squared error for one training pattern
    void backProp(double target);            // Calculate error and propagate deltas back
    void updateWeights();                   // Update weights
    double activate(double S);                // Sigmoid activation function (logistic)
    double trainOne(vector<double> tuple);    // Feeds a single tuple through network and adjusts erros accordingly
    double runOne(vector<double> tuple);     // Tests a single tuple

public:
    MultilayerNN(int _inputNodes, int _hiddenNodes, int _hiddenLayers, int _outputNodes, string actFunc,
                                   string _activateOutput, double momentum, double learningRate);
    MultilayerNN(const MultilayerNN& orig);
     void reset();
     ~MultilayerNN();
     vector<double> train(vector<vector<double>> tset);
     double run(vector<vector<double>> data);
     string const className()  { return "MLP"; }

};

#endif	/* MULTILAYERNN_H */

