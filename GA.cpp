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
        for (int i = 1; i <= children_parent_ratio * 50; i++) {
            // Randomly select two parents with uniform probability
            //TODO change these two lines
            Chromosome p1 = population.at(dist(rd));
            Chromosome p2 = population.at(dist(rd));
            // Create child via recombination
            offspring.push_back(crossover(p1, p2)); //TODO make sure this part works when returning a pointer
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

        // Select survivors
        // Create combined parent/offspring set
        //TODO change this stuff
        selectionChroms.reserve(population.size() + offspring.size()); // preallocate memory
        selectionChroms.insert(selectionChroms.end(), population.begin(), population.end());
        selectionChroms.insert(selectionChroms.end(), offspring.begin(), offspring.end());

        // Clear population
        population.clear();

        // Pull out the first 50 to replace population
        for (int i = 0; i < 50; i++) {
            // Iterator pointing to next min
            auto it = min_element(selectionChroms.begin(), selectionChroms.end());

            // Save this element, then erase it from vector
            currentMin = *it;
            selectionChroms.erase(it);

            //TODO this is the stopping step
            // If this is absolute minimum, save it as current best
            if (i == 0) {
                if (currentMinimumError - currentMin.nn.lastMSE < 0.001) lowDeltaCounter++;
                else lowDeltaCounter = 0;
                currentMinimumError = currentMin.nn.lastMSE;
            }

            // Push this element to popualtion
            population.push_back(currentMin);

            //print
            //cout << currentMin.nn.lastMSE << endl;
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

GA::Chromosome* GA::crossover(Chromosome p1, Chromosome p2) {
    Chromosome two_children[2] = {p1,p2};
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