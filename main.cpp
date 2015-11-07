#include <iostream>
#include "Experiment.h"
#include "Updater.h"
#include "ES.h"
#include <vector>

using namespace std;

int main() {
    vector <Updater*> u;

    ES es(7, 50000, 0.001, 6, 10, 5, 2, "sigmoid", "sigmoid");

    u.push_back(&es);

    Experiment e(u, {"bupa_n.csv", "ecoli_n.csv"}, 200, 7);

    e.runExperiment();
}