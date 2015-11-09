//
// Created by Fred Vollmer on 9/29/15.
//

#ifndef GIT_EXPERIMENT_H
#define GIT_EXPERIMENT_H

#include <vector>
#include "Updater.h"
#include <string>
#include <sstream>
#include <queue>

using namespace std;

class Experiment {

    vector<Updater*> updaters;    // The algorithm object to run excperiemnt on
    int columns;     // Number of x for Rosenbrock function
    int rows;          // Sample size

    vector<vector<double>> dataset;
    vector<vector<double>> trainingData;
    vector<vector<double>> testingData;
    vector<vector<double>> doubleInputData;
    vector<double> doubleOutputData;
    vector<vector<double>> doubleTrainingInput;
    vector<double> doubleTrainingOutput;
    vector<vector<double>> doubleTestingInput;
    vector<double> doubleTestingOutput;
    vector<string> datasets;
    string datasetName;

public:
    Experiment(vector<Updater *> _a, vector<string> _datasets, int rows, int columns);
    bool runExperiment();
    void getDoubleData();
private:
    void getData(string dataPath);
    void readData(string file);
    void nextFold();                            // For re-folding
    void nextIteration();                       // For next iteration of CV
    void printMatrix(vector<vector<double>> v);  // Helper method for outputting dataset

    // Helper function for converting string to double
    double stringToNumber ( const string &Text )
    {
        istringstream ss(Text);
        double result;
        return ss >> result ? result : 0;
    }

};


#endif //GIT_EXPERIMENT_H
