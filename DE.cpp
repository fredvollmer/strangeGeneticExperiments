//
// Created by Austo89 on 11/3/2015.
//

#include <random>
#include "DE.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <float.h>

// Constructor

DE::DE(double _beta, double _crossover_prob, int _pop_size,int _max_generations, double _targetError, int _inputNodes,
       int _hiddenNodes, int _hiddenLayers, int _outputNodes, string _activateHidden, string _activateOutput){
    beta = _beta;
    crossover_prob = _crossover_prob;
    pop_size = _pop_size;
    max_generations = _max_generations;
    targetMSE = _targetError;
    inputNodesN = _inputNodes;
    hiddenNodesPerLayer = _hiddenNodes;
    hiddenLayerCount = _hiddenLayers;
    outputNodesN = _outputNodes;
    hiddenActivation = _activateHidden;
    outputActivation = _activateOutput;

}

MultilayerNN DE::train(vector<vector<double>>* _dataset){
    random_device rd;
    double min_err = DBL_MAX;
    double target_err = .0001;
    MultilayerNN bestNetwork;

    dataset = *_dataset;

    cur_generation = 0;
    initPopulation(pop_size);
    bestNetwork = networks[0];

    fstream resultStream;
    resultStream.open("run_DE.csv", ofstream::out | ofstream::trunc);
    // Check for stream error
    if (resultStream.fail()) {
        cerr << "open stream failure at rs: " << strerror(errno) << '\n';
    }

    while (cur_generation < max_generations && min_err > target_err){
        for(int i = 0; i < pop_size; i++){
            //mutation
            vector<vector<double>> trial_weights = createTrialVector(i);

            //crossover
            crossover(i, trial_weights);

            //compare fitness
            double old_fit = fitnessFunction(networks[i]);
            vector<vector<double>> oldWeights = networks[i].weights;
            networks[i].weights = trial_weights;
            double new_fit = fitnessFunction(networks[i]);

            //switch back if the trial vector is worse
            if (old_fit < new_fit){
                new_fit = old_fit;
                networks[i].weights = oldWeights;
            }

            //check if the new network is the best
            if(new_fit < min_err ){
                min_err = new_fit;
                bestNetwork = networks[i];
            }
        }
        // Output result every 50 gens
        if (generation % 50 == 0) {
            resultStream << generation << "," << min_err << endl;
        }

        //increment the generation
        cur_generation++;
    }
}

void DE::crossover(int cur_index, vector<vector<double>> trial) {
    uniform_real_distribution<double> crossover_chance(0.0, 1.0);
    default_random_engine generator;
    vector<vector<double>> aWeights = networks[cur_index].weights;
    MultilayerNN vecX = networks[cur_index];

    //iterate over every weight
    for(int i = 0; i < aWeights.size(); i++){
        for (int j = 0; j < aWeights[i].size(); j++){
            //roll the dice uniformly, creates a double between (0, 1)
            double cr = crossover_chance(generator);

            // binomial crossover
            if (cr > crossover_prob){
                //grab the weight back from the x vector
                trial[i][j] = vecX.weights[i][j];
            }
        }
    }
}

vector<vector<double>> DE::createTrialVector(int cur_index) {
    random_device rd;
    uniform_real_distribution<double> beta_size(beta * -1, beta);
    default_random_engine generator;
    int rand_indexB, rand_indexC;
    bool bFlag = true;
    bool cFlag = true;

    //create a random scaling value between (-beta, beta) uniformly
    double cur_beta = beta_size(generator);

    // pick 2 data points at random that are different from each other
    // and the current data point
    while(bFlag){
        rand_indexB = rd()% pop_size;
        if (cur_index != rand_indexB){
            bFlag = false;
        }
    }
    while(cFlag){
        rand_indexC = rd()% pop_size;
        if (cur_index != rand_indexC && rand_indexB != rand_indexC){
            cFlag = false;
        }
    }

    // retrieve the weights
    vector<vector<double>> trialWeights;
    vector<vector<double>> aWeights = networks[cur_index].weights;
    vector<vector<double>> bWeights = networks[rand_indexB].weights;
    vector<vector<double>> cWeights = networks[rand_indexC].weights;

    // loop over the weights and update them
    for(int i = 0; i < aWeights.size(); i++){
        for (int j = 0; j < aWeights[i].size(); j++){
            double new_weight = aWeights[i][j] + cur_beta * (bWeights[i][j] - cWeights[i][j]);
            trialWeights[i].push_back(new_weight);
        }
    }

    // return the weights after crossover
    return trialWeights;
}

double DE::fitnessFunction(MultilayerNN net) {
    // run the network with the current weights to
    // obtain the mean squared error
    net.run(dataset);
    return net.lastMSE;
}

void DE::initPopulation(int population_size) {
    // create the networks for the population
    for (int i = 0; i < population_size; i++) {
        MultilayerNN nn(inputNodesN, hiddenNodesPerLayer,
                        hiddenLayerCount, outputNodesN, hiddenActivation, outputActivation);
        networks.push_back(nn);
    }
}