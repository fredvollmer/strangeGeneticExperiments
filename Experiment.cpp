//
// Created by Fred Vollmer on 9/29/15.
//

#include "Experiment.h"
#include "MultilayerNN.h"
#include "Updater.h"
#include <unordered_set>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>

Experiment::Experiment(vector<Updater *> _a, string _dataset, int _rows, int _columns) {
    // Set parameters
    updaters = _a;
    columns = _columns;
    rows = _rows;

    // Load dataset
    getData(_dataset);
}

void Experiment::nextIteration() {
    // Clear out training and testing data
    trainingData.clear();
    testingData.clear();

    unordered_set<int> selectedIndices;
    int n = dataset.size() / 2;
    int max = dataset.size();
    random_device rd;
    uniform_int_distribution<int> dist(0, max - 1);

    while (selectedIndices.size() < n) {
        selectedIndices.insert(dist(rd));
    }
    // Add each tuple to proper set set
    for (int i = 0; i < dataset.size(); i++) {
        if (selectedIndices.count(i) > 0) {
            trainingData.push_back(dataset.at(i));
        } else {
            testingData.push_back(dataset.at(i));
        }
    }
}

// Flip training and testing data
void Experiment::nextFold() {
    vector<vector<double>> temp;
    temp = testingData;
    testingData = trainingData;
    trainingData = temp;
}

void Experiment::getData(string dataPath) {
    ifstream dataStream;
    string s_cell;
    char * cell;

    // Open data file for reading
    dataStream.open(dataPath);
    if (dataStream.fail()) {                    // Check for stream error
        cerr << "Read stream failure: " << strerror(errno) << '\n';
    }
    // Loop through each tuple
    for (int t = 0; t < rows; t = t + 1) {
        vector<double> newTuple;                // Create empty vector
        dataset.push_back(newTuple);            // Add new tuple to dataset
        getline(dataStream, s_cell);              // Read next "block" from data file
        cell = &s_cell[0u];

        cell = strtok(cell, ",");               // Tokenize cell by ','

        // Loop through each input, plus output
        for (int n = 0; n < columns; n = n + 1) {
            double num = stringToNumber(cell);  // Convert cell string to double
            dataset.at(t).push_back(num);       // Add this cell to dataset
            cell  = strtok(NULL, ",");          // Next token
        }
    }

    printMatrix(dataset);

    dataStream.close();

}

void Experiment::printMatrix(vector<vector<double>> v) {
    vector<vector<double> >::iterator row;
    vector<double>::iterator col;
    for (row = v.begin(); row != v.end(); row++) {
        for (col = row->begin(); col != row->end(); col++) {
            cout << *col << " ";
        }
        cout << "\n";
    }
}

bool Experiment::runExperiment() {

    fstream resultStream;
    resultStream.open("experiemntResults.txt", ofstream::out | ofstream::trunc);
    // Check for stream error
    if (resultStream.fail()) {
        cerr << "open stream failure at rs: " << strerror(errno) << '\n';
    }

    //nextIteration();
    //a.front() -> train(trainingData);

    // 5X
    for (int i = 0; i < 5; i++) {
        nextIteration();
        // 2 CV
        for (int j = 0; j < 2; j++) {
            nextFold();

            // Train all algorithms, run their returned NN
            for (int alg = 0; alg < updaters.size(); alg++) {
                MultilayerNN resultingNN = updaters[alg]->train(&trainingData);
                resultStream << i << "," << j << "," << updaters[alg]->nickname() <<
                "," << resultingNN.run(testingData) << endl;
            }
        }
    }

    // Close writers
    resultStream.close();

    return true;
}