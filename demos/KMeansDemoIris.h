//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/KMeans.h"
#include "../include/ToyDataSets.h"

void demoKMeansIris(unsigned nClusters, unsigned initMethod) {

    metis::DataLabeled *iris = metis::loadIris();

    double before = omp_get_wtime();
    metis::KMeans clusterer(nClusters, initMethod);
    clusterer.cluster(iris->getInputs());
    double after = omp_get_wtime();

    std::cout << "\nInput has been clustered with centroids:" << std::endl;
    std::cout << clusterer.getCentroids() << std::endl;

    std::cout << "\nMean distance:" << std::endl;
    std::cout << clusterer.score(iris->getInputs()) << std::endl;

    std::cout << "\nPredictions:" << std::endl;
    std::cout << clusterer.getLabels().transpose() << std::endl;

    std::cout << "\nComputation of clusters took:" << std::endl;
    std::cout << 1000 * (after - before) << "ms" << std::endl;

}