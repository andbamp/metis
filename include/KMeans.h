//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_KMEANS_H
#define METIS_KMEANS_H


#include "DataSet.h"

namespace metis {

class KMeans {

private:

    // 3. Structure
    Eigen::MatrixXd _centroids;
    Eigen::VectorXi _labels;

    // 4. State
    unsigned _nClusters = 5;
    unsigned _initMethod = 1;
    unsigned _maxIterations = 500;

    unsigned _nDimensions;
    unsigned _nInstances;

    // 5. Helper methods
    void initializeCentroidsRandom();
    void initializeCentroidsKMeansPP();
    void redetermineCentroids();

    unsigned closestCentroid(Eigen::VectorXd instance) const;

public:

    // 2. Interface methods
    Eigen::VectorXi cluster(DataSet *data);

    Eigen::VectorXi predict(DataSet *data) const;
    Eigen::VectorXi predict(Eigen::MatrixXd *data) const;

    double score(DataSet *data) const;

    Eigen::MatrixXd getCentroids() const;
    Eigen::VectorXi getLabels() const;

    void setNumberOfClusters(unsigned nClusters);
    void setInitializationMethod(unsigned initMethod);
    void setMaxIterations(unsigned iterations);

    // 1. Construction
    KMeans(unsigned nClusters, unsigned initMethod);
    KMeans(unsigned nClusters);
    KMeans();
    ~KMeans();

};

}


#endif //METIS_KMEANS_H
