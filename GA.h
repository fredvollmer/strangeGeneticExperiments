//
// Created by Fred Vollmer on 11/1/15.
//

#ifndef ASSIGNMENT3_GA_H
#define ASSIGNMENT3_GA_H


#include "Updater.h"
#include <vector>

class GA : public Updater {

public:
    // Constructor
    GA(int _maxGenerations, double _targetError, int _inputNodes, int _hiddenNodes,
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
    void selection(Chromosome parents[2]);
    Chromosome* crossover(Chromosome* p, Chromosome* two_children);
    void mutate(Chromosome &c);
    void populationSetup();
};

#endif //ASSIGNMENT3_GA_H
