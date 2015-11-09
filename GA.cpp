//
// Created by Fred Vollmer on 11/1/15.
//

#include "GA.h"
#include "float.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm>

GA::GA(int _maxGenerations, double _targetError, int _inputNodes, int _hiddenNodes,
       int _hiddenLayers, int _outputNodes,
       string _activateHidden, string _activateOutput) {

    // Assign ivars
    maxGenerations = _maxGenerations;
    targetMSE = _targetError;
    inputNodesN = _inputNodes;
    hiddenNodesPerLayer = _hiddenNodes;
    hiddenLayerCount = _hiddenLayers;
    outputNodesN = _outputNodes;
    hiddenActivation = _activateHidden;
    outputActivation = _activateOutput;

    // Assign learning rates heuristically
    overallLearningRate = 1 / pow(2 * 50, 0.5);
    cwLearningRate = 1 / pow(2 * pow(50, 0.5), 0.5);

}

MultilayerNN GA::train(vector<vector<double>> *_dataset) {
    int lowDeltaCounter = 0;
    random_device rd;                                               // Initialize random device & distribution
    uniform_int_distribution<u_long> dist(0, 50 - 1);
    normal_distribution<double> norm(0, 1);
    double currentMinimumError = DBL_MAX;
    vector<Chromosome> selectionChroms(50 * children_parent_ratio + 50);
    Chromosome currentMin;

    fstream resultStream;
    resultStream.open("run_GA.csv", ofstream::out | ofstream::trunc);
    // Check for stream error
    if (resultStream.fail()) {
        cerr << "open stream failure at rs: " << strerror(errno) << '\n';
    }

    // Save dataset
    dataset = *_dataset;

    // Init population for this training run
    populationSetup();

    // Initial network run to get errors
    runNetworks();

    // Generation loop
    // Continue until max gens. reached or error reduces to threshold
    while (generation <= maxGenerations && targetMSE < currentMinimumError && lowDeltaCounter < 100) {

        // Clear offspring
        offspring.clear();
        selectionChroms.clear();

        // Generate offspring, add into temporary offspring pool
        for (int i = 0; i < population.size()/2; i++) {
            // select the best two parents out of ten random chromosomes
            random_shuffle(population.begin(), population.end());
            Chromosome parents[2];
            selection(parents);
            // Create child via recombination
            Chromosome *children;
            children = crossover(parents);
            offspring.push_back(children[0]);
            offspring.push_back(children[1]);
        }
        if (population.size()%2==1){
            // select the best two parents out of ten random chromosomes
            random_shuffle(population.begin(), population.end());
            Chromosome parents[2];
            selection(parents);
            // Create child via recombination
            Chromosome *children;
            children = crossover(parents);
            offspring.push_back(children[0]);
        }

        // Overall random number from N(0,1) for this generation
        //double globalTerm = norm(rd);

        // Mutate offspring
        for (auto &c : offspring) {
            mutate(c);
        }

        // Run offspring networks
        for (int i = 0; i < offspring.size(); i++) {
            offspring[i].nn.run(dataset);
        }

        // Clear population
        population.clear();

        population=offspring;

        //find minimum error
        for (int i = 0; i < population.size(); i++) {
                if(population[i].nn.lastMSE<currentMinimumError) {
                    currentMinimumError=population[i].nn.lastMSE;
                }
        }

        // Output result every 50 gens
        if (generation % 50 == 0) {
            resultStream << generation << "," << currentMinimumError << endl;
        }

        cout << "Generation " << generation << ": " << currentMinimumError << endl;
        cout << lowDeltaCounter << endl;

        // Next generation
        generation++;
    }

    resultStream.close();

    return currentMin.nn;
}

void GA::runNetworks() {
    for (int i = 0; i < population.size(); i++) {
        population[i].nn.run(dataset);
    }
}

void GA::selection(Chromosome parents[2]) {
    //get the first 10 chromosomes and use select the best
    parents[0]=population[0];
    parents[1]=population[1];

    // Create child network by taking average of each weight from parents 7t80
    for (int i = 1; i < 10; i++) {
        if(parents[0].nn.lastMSE>population[i].nn.lastMSE){
            parents[1]=parents[0];
            parents[0]=population[i];
        }
    }
}

GA::Chromosome* GA::crossover(Chromosome* p) {
    Chromosome two_children[2] = {p[0],p[1]};
    double temp;

    // Create child network by taking average of each weight from parents 7t80
    for (int i = 0; i < two_children[0].nn.weights.size(); i++) {
        for (int j = 0; j < two_children[0].nn.weights[i].size(); j++) {
            //crossover the individual value a 5th of the time
            if(rand()>__RAND_MAX/5){
                temp=two_children[0].nn.weights[i][j];
                two_children[0].nn.weights[i][j]=two_children[1].nn.weights[i][j];
                two_children[1].nn.weights[i][j]=temp;
            }
        }
    }

    // Child step size is average of parents'
    /*child.stepSize = (p1.stepSize + p2.stepSize) / 2;*/

    return &two_children[0];
}

void GA::mutate(Chromosome &c) {
    normal_distribution<double> norm(0, .2);
    random_device rd;
    // Mutate step size
    /*double delta = exp((overallLearningRate * globalTerm) + (cwLearningRate * norm(rd)));
    c.stepSize = c.stepSize * delta;*/

    // Mutate object function: mutate each weight
    for (int i = 0; i < c.nn.weights.size(); i++) {
        for (int j = 0; j < c.nn.weights[i].size(); j++) {
            if (rand()>__RAND_MAX/2) {
                c.nn.weights[i][j] += c.nn.weights[i][j] * norm(rd);
            }
        }
    }
}

void GA::populationSetup() {

    generation = 0;
    population.clear();

    // init networks
    initPopulation();

    // Init population, initial step size is 1
    /*for (auto &nn : networks) {
        Chromosome *p = new Chromosome();
        p->nn = nn;
        p->stepSize = 1;
        population.push_back(*p);
    }*/
}