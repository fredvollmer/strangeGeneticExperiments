#include <iostream>
#include "Experiment.h"
#include "Updater.h"
#include "ES.h"
#include "DE.h"
#include <vector>

using namespace std;

int main() {
    // vector for updater method.
    // originally planned to set up all tests to run at once, but it didn't
    // pan out. Ran them individually.
    vector <Updater*> u;

    // create an updater subclass object to run the test

    //ES es(7, 50000, 0.001, 6, 10, 5, 2, "sigmoid", "sigmoid");
    DE de(2.0,0.2,100,100000,.0001,7,16,3,9,"sigmoid","sigmoid");

    // push the updater onto the somewhat ancillary updater vector
    u.push_back(&de);

    // create the experiment object that will generate the test runs and auto perform
    // 5x2 cross validation
    Experiment e(u, {"C:\\Users\\Austo89\\Documents\\Homework\\Fall 2015\\Soft Computing\\project3\\datasets norm\\ecoli_n.csv"}, 100, 7);

    e.runExperiment();
}