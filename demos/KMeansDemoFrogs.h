//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/KMeans.h"
#include "../include/ToyDataSets.h"

void demoKMeansFrog(unsigned nClusters, unsigned initMethod, unsigned nThreads) {

    omp_set_num_threads(nThreads);

    // Creating data set
    std::string myPath = "../data/other/Frogs_MFCCs.csv";
    metis::DataSet data(myPath, ',');

    std::vector<unsigned> attr;
    for (unsigned a = 0; a < 21; ++a) attr.push_back(a);
    data.create(attr);

    // Clustering
    double before = omp_get_wtime();
    metis::KMeans clusterer(nClusters, initMethod);
    clusterer.cluster(&data);
    double after = omp_get_wtime();

    // Results
    std::cout << "\nInput has been clustered with centroids:" << std::endl;
    std::cout << clusterer.getCentroids() << std::endl;

    std::cout << "\nMean distance:" << std::endl;
    std::cout << clusterer.score(&data) << std::endl;

    std::cout << "\nPredictions:" << std::endl;
    std::cout << clusterer.getLabels().transpose() << std::endl;

    std::cout << "\nComputation of clusters took:" << std::endl;
    std::cout << 1000 * (after - before) << "ms" << std::endl;

}