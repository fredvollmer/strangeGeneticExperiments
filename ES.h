//
// Created by Fred Vollmer on 11/1/15.
//

#ifndef ASSIGNMENT3_ES_H
#define ASSIGNMENT3_ES_H


#include "Updater.h"
#include <vector>

class ES : public Updater {

public:
    // Constructor
    ES(int _children_parent_ratio, int _maxGenerations, double _targetError, int _inputNodes, int _hiddenNodes,
       int _hiddenLayers, int _outputNodes,
       string _activateHidden, string _activateOutput);
    // Extending virtual functions
    MultilayerNN train(vector<vector<double>>* _dataset) override;
    string nickname() override { return "Evolution Strategy"; }

private:

    // Population chromosome struct
    struct Chromosome {
        MultilayerNN nn = MultilayerNN();
        double stepSize = 0;
        Chromosome() {};
        // Lamda func for sorting by network error
        bool operator< (const Chromosome &other) const {
            return nn.lastMSE < other.nn.lastMSE;
        }
    };

    vector<Chromosome> population;
    int children_parent_ratio;
    double overallLearningRate;
    double cwLearningRate;
    int maxGenerations;
    int generation = 0;
    vector<Chromosome> offspring;
    vector<double> populationErrors;
    void runNetworks();
    Chromosome recombination(Chromosome p1, Chromosome p2);
    void mutate(Chromosome &c, double globalTerm);
};

#endif //ASSIGNMENT3_ES_H
