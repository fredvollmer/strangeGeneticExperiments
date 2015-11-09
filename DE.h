//
// Created by Austo89 on 11/3/2015.
//

#include "Updater.h"
#include <vector>
#include <math.h>
#include "MultilayerNN.h"

using namespace std;

#ifndef ASSIGNMENT3_DE_H
#define ASSIGNMENT3_DE_H


class DE: public Updater {
private:
    vector<vector<double>> population;
    double beta;
    double crossover_prob;
    int pop_size;
    int generation;
    int dimensions;
    int max_generations;
    int cur_generation;
    vector<vector<double>> dataset;

    void crossover(int cur_index, vector<vector<double>> trial);
    vector<vector<double>> createTrialVector(int cur_index);
    double fitnessFunction(MultilayerNN net);
    void initPopulation(int population_size);

public:
    DE(double _beta, double _crossover_prob, int _pop_size,int _max_generations, double _targetError, int _inputNodes,
       int _hiddenNodes, int _hiddenLayers, int _outputNodes, string _activateHidden, string _activateOutput);
    MultilayerNN train(vector<vector<double>>* _dataset) override;
    string nickname() override { return "Differential Evolution"; }
};


#endif //ASSIGNMENT3_DE_H
