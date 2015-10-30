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

private:
    vector<vector<double>> dataset;
};


#endif //ASSIGNMENT3_LEARNER_H
