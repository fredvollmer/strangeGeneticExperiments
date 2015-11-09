#include <iostream>
#include "Experiment.h"
#include "Updater.h"
#include "ES.h"
#include "DE.h"
#include <vector>

using namespace std;

int main() {
    vector <Updater*> u;

    //ES es(7, 50000, 0.001, 6, 10, 5, 2, "sigmoid", "sigmoid");
    DE de(2.0,0.2,100,100000,.0001,7,16,3,9,"sigmoid","sigmoid");

    u.push_back(&de);

    Experiment e(u, {"C:\\Users\\Austo89\\Documents\\Homework\\Fall 2015\\Soft Computing\\project3\\datasets norm\\ecoli_n.csv"}, 100, 7);

    e.runExperiment();
}