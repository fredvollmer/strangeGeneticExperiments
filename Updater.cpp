//
// Created by Fred Vollmer on 10/28/15.
//

#include "Updater.h"
#include "MultilayerNN.h"

void Updater::initPopulation() {
    for (int i = 0; i < 50; i++) {
        MultilayerNN nn(inputNodesN, hiddenNodesPerLayer,
                        hiddenLayerCount, outputNodesN, hiddenActivation, outputActivation);
        networks.push_back(nn);
    }
}