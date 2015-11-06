//
// Created by Austo89 on 11/3/2015.
//

#include "Updater.h"
#include <vector>
#include <math.h>

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

public:
    DE(double _beta, double _crossover_prob, int _pop_size);
    MultilayerNN train(vector<vector<double>>* _dataset) override;
    string nickname() override { return "Differential Evolution"; }
};


#endif //ASSIGNMENT3_DE_H
