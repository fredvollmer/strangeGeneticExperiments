//
// Created by Fred Vollmer on 10/28/15.
//

/**
 * Provides an interface for all weight update algorithms
 */

#ifndef ASSIGNMENT3_LEARNER_H
#define ASSIGNMENT3_LEARNER_H

#include <vector>
#include "MultilayerNN.h"

using namespace std;

class Updater {
public:
    // Pure virtual functions
    virtual MultilayerNN train(vector<vector<double>>* _dataset) = 0;
    virtual string nickname() = 0;

protected:
    vector<vector<double>> dataset;
    vector<MultilayerNN> networks;

    // Neural network proerties
    int inputNodesN;
    int outputNodesN;
    int hiddenLayerCount;
    int hiddenNodesPerLayer;
    double targetMSE;
    string outputActivation;
    string hiddenActivation;

    void initPopulation();
};


#endif //ASSIGNMENT3_LEARNER_H
