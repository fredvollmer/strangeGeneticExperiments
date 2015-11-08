//
// Created by Austo89 on 11/3/2015.
//

#include <bits/random.h>
#include "DE.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string.h>

DE::DE(double _beta, double _crossover_prob, int _pop_size,int _max_generations, double _targetError, int _inputNodes,
       int _hiddenNodes, int _hiddenLayers, int _outputNodes, string _activateHidden, string _activateOutput){
    beta = _beta;
    crossover_prob = _crossover_prob;
    pop_size = _pop_size;
    max_generations = _max_generations;
}

MultilayerNN DE::train(vector<vector<double>>* _dataset){
    random_device rd;
    uniform_real_distribution<double> crossover_chance(0.0, 1.0);
    uniform_real_distribution<double> beta_size(beta * -1, beta);
    double min_err;
    double target_err = .0001;

    dataset = *_dataset;

    cur_generation = 0;
    initPopulation(pop_size);

    fstream resultStream;
    resultStream.open("run_DE.csv", ofstream::out | ofstream::trunc);
    // Check for stream error
    if (resultStream.fail()) {
        cerr << "open stream failure at rs: " << strerror(errno) << '\n';
    }

    while (cur_generation < max_generations && min_err > target_err){

    }
}

void DE::crossover() {

}

vector<double> DE::createTrainingVector() {

}

double DE::fitnessFunction() {

}

void DE::initPopulation(int population_size) {
    for (int i = 0; i < population_size; i++) {
        MultilayerNN nn(inputNodesN, hiddenNodesPerLayer,
                        hiddenLayerCount, outputNodesN, hiddenActivation, outputActivation);
        networks.push_back(nn);
    }
}