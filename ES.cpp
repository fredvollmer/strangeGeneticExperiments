//
// Created by Fred Vollmer on 11/1/15.
//

#include "ES.h"
#include "float.h"
#include <random>
#include <iostream>
#include <fstream>
#include <string.h>

#include <bits/stl_algo.h>

#include <algorithm>


ES::ES(int _children_parent_ratio, int _maxGenerations, double _targetError, int _inputNodes, int _hiddenNodes,
       int _hiddenLayers, int _outputNodes,
       string _activateHidden, string _activateOutput) {

    // Assign ivars
    children_parent_ratio = _children_parent_ratio;
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

MultilayerNN ES::train(vector<vector<double>> *_dataset) {
    int lowDeltaCounter = 0;
    random_device rd;                                               // Initialize random device & distribution
    uniform_int_distribution<u_long> dist(0, 50 - 1);
    normal_distribution<double> norm(0, 1);
    double currentMinimumError = DBL_MAX;
    MultilayerNN currentMinimumNetwork;
    vector<Chromosome> selectionChroms(50 * children_parent_ratio + 50);
    Chromosome currentMin;

    fstream resultStream;
    resultStream.open("run_ES.csv", ofstream::out | ofstream::trunc);
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
            Chromosome p1 = population.at(dist(rd));
            Chromosome p2 = population.at(dist(rd));
            // Create child via recombination
            offspring.push_back(recombination(p1, p2));
        }

        // Overall random number from N(0,1) for this generation
        double globalTerm = norm(rd);

        // Mutate offspring
        for (auto &c : offspring) {
            mutate(c, globalTerm);
        }

        // Run offspring networks
        for (int i = 0; i < offspring.size(); i++) {
            offspring[i].nn.run(dataset);
        }

        // Select survivors
        // Create combined parent/offspring set
        selectionChroms.reserve(population.size() + offspring.size()); // preallocate memory
        selectionChroms.insert(selectionChroms.end(), population.begin(), population.end());
        selectionChroms.insert(selectionChroms.end(), offspring.begin(), offspring.end());

        // Clear population
        population.clear();

        // Pull out the first 50 to replace population
        for (int i = 0; i < 50; i++) {
            // Iterator pointing to next min
//<<<<<<< HEAD
//            //it = min_element(selectionChroms.begin(), selectionChroms.end());
//
//            // Save this element, then erase it from vector
//            currentMin = selectionChroms.at(it);
//            //selectionChroms.erase(it);
//=======
            auto it = min_element(selectionChroms.begin(), selectionChroms.end());

            // Save this element, then erase it from vector
            currentMin = *it;
            selectionChroms.erase(it);
//>>>>>>> 0a88efe32cc5835371bb086ddf1c8d9ac4be49cb

            // If this is absolute minimum, save it as current best
            if (i == 0) {
                if (currentMinimumError - currentMin.nn.lastMSE < 0.001) lowDeltaCounter++;
                else lowDeltaCounter = 0;
                currentMinimumError = currentMin.nn.lastMSE;
                currentMinimumNetwork = currentMin.nn;
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

    return currentMinimumNetwork;
}

void ES::runNetworks() {
    for (int i = 0; i < population.size(); i++) {
        population[i].nn.run(dataset);
    }
}

ES::Chromosome ES::recombination(Chromosome p1, Chromosome p2) {
    // Intermediate recombination with r = 2
    // Copy p1 to be child...we'll replace its weights next
    Chromosome child = p1;

    // Create child network by taking average of each weight from parents
    for (int i = 0; i < child.nn.weights.size(); i++) {
        for (int j = 0; j < child.nn.weights[i].size(); j++) {
            child.nn.weights[i][j] = (p1.nn.weights[i][j] + p2.nn.weights[i][j]) / 2;
        }
    }

    // Child step size is average of parents'
    child.stepSize = (p1.stepSize + p2.stepSize) / 2;

    return child;
}

void ES::mutate(Chromosome &c, double globalTerm) {
    normal_distribution<double> norm(0, 1);
    random_device rd;
    // Mutate step size
    double delta = exp((overallLearningRate * globalTerm) + (cwLearningRate * norm(rd)));
    c.stepSize = c.stepSize * delta;

    // Mutate object function: mutate each weight
    for (int i = 0; i < c.nn.weights.size(); i++) {
        for (int j = 0; j < c.nn.weights[i].size(); j++) {
            c.nn.weights[i][j] += c.stepSize * norm(rd);
        }
    }
}

void ES::populationSetup() {

    generation = 0;
    population.clear();

    // init networks
    initPopulation();

    // Init population, initial step size is 1
    for (auto &nn : networks) {
        Chromosome *p = new Chromosome();
        p->nn = nn;
        p->stepSize = 1;
        population.push_back(*p);
    }
}