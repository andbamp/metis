//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/KMeans.h"
#include "omp.h"

namespace {

    Eigen::MatrixXd *g_data;

    unsigned g_nThreads;

    Eigen::VectorXd *g_distances;
    Eigen::MatrixXd *g_newCentroids;
    std::vector<unsigned> *g_newClusterSizes;

}

void metis::KMeans::initializeCentroidsRandom() {

    rand();
    unsigned point;
    Eigen::VectorXd centroid;

    bool repeat;
    for (unsigned c = 0; c < _nClusters; ++c) {
        repeat = true;
        while (repeat) {

            point = ((double) rand() / (RAND_MAX)) * _nInstances;
            centroid = g_data->row(point);

            repeat = false;
            for (unsigned r = 0; r < c; ++r)
                if (centroid.isApprox(_centroids.row(r).transpose()))
                    repeat = true;

        }

        _centroids.row(c) = centroid;
    }

}

void metis::KMeans::initializeCentroidsKMeansPP() {

    rand();

    unsigned point;
    Eigen::VectorXd centroid;
    std::vector<Eigen::VectorXd> centroids;

    Eigen::VectorXd distances(_nInstances);
    Eigen::VectorXd weights(_nInstances);
    double distance;
    double weight;

    point = ((double) rand() / (RAND_MAX)) * _nInstances;
    centroid = g_data->row(point);
    centroids.push_back(centroid);

    for (unsigned i = 0; i < _nInstances; ++i)
        distances.coeffRef(i) = (centroids[0].transpose() - g_data->row(i)).squaredNorm();

    while (centroids.size() < _nClusters) {

        distances.array() /= distances.sum();
        weights.setZero();
        for (unsigned w = 1; w < _nInstances; ++w)
            weights.coeffRef(w) = weights.coeffRef(w-1) + distances.coeffRef(w-1);

        weight = (double) rand() / (RAND_MAX);

        point = 0;
        for (unsigned w = 1; w < _nInstances; ++w) {
            if (weight > weights.coeffRef(w)) ++point;
            else break;
        }

        centroid = g_data->row(point);
        centroids.push_back(centroid);

        for (unsigned i = 0; i < _nInstances; ++i) {

            distance = (centroid.transpose() - g_data->row(i)).squaredNorm();
            if (distance < distances.coeffRef(i))
                distances.coeffRef(i) = distance;

        }

    }

    for (unsigned c = 0; c < _nClusters; ++c)
        _centroids.row(c) = centroids[c];

}

unsigned metis::KMeans::closestCentroid(Eigen::VectorXd instance) const {

    unsigned th = omp_get_thread_num();
    g_distances[th] = (_centroids.rowwise() - instance.transpose()).rowwise().squaredNorm();

    unsigned nearestCluster = 0;
    double leastDistance = g_distances[th].coeffRef(0);
    for (unsigned c = 1; c < _nClusters; ++c) {
        if (g_distances[th].coeffRef(c) < leastDistance) {
            nearestCluster = c;
            leastDistance = g_distances[th].coeffRef(c);
        }
    }

    return nearestCluster;

}

void metis::KMeans::redetermineCentroids() {

#pragma omp parallel
    {
        unsigned th = omp_get_thread_num();

        g_newCentroids[th].setZero();

        g_newClusterSizes[th].clear();
        g_newClusterSizes[th].resize(_nClusters);

        unsigned selectedCluster;

#pragma omp for
        for (unsigned i = 0; i < _nInstances; ++i) {
            selectedCluster = closestCentroid(g_data->row(i));
            g_newCentroids[th].row(selectedCluster) += g_data->row(i);
            g_newClusterSizes[th][selectedCluster] += 1;
        }
    };

    for (unsigned t = 1; t < g_nThreads; ++t) {
        for (unsigned c = 0; c < _nClusters; ++c) {
            g_newCentroids[0].row(c) += g_newCentroids[t].row(c);
            g_newClusterSizes[0][c] += g_newClusterSizes[t][c];
        }
    }

    for (unsigned c = 0; c < _nClusters; ++c) {
        if (g_newClusterSizes[0][c] == 0) {
            g_newCentroids[0].row(c) = _centroids.row(c);
        } else
            g_newCentroids[0].row(c) /= g_newClusterSizes[0][c];
    }

}

Eigen::VectorXi metis::KMeans::cluster(metis::DataSet *data) {

    _nDimensions = data->attributes();
    _nInstances = data->instances();

    _centroids.resize(_nClusters, _nDimensions);
    _labels.resize(_nInstances);

    g_data = data->getData();

#pragma omp parallel
    {
        g_nThreads = omp_get_num_threads();
    };

    g_distances = new Eigen::VectorXd[g_nThreads];
    g_newCentroids = new Eigen::MatrixXd[g_nThreads];
    g_newClusterSizes = new std::vector<unsigned>[g_nThreads];

#pragma omp parallel
    {
        unsigned th = omp_get_thread_num();
        g_distances[th].resize(_nClusters);
        g_newCentroids[th].resize(_nClusters, _nDimensions);
        g_newClusterSizes[th].resize(_nClusters);
    };

    switch (_initMethod) {
        case 0:
            initializeCentroidsRandom();
            break;
        case 1:
            initializeCentroidsKMeansPP();
            break;
        default:
            initializeCentroidsKMeansPP();
            break;
    }

    unsigned i;
    for (i = 0; i < _maxIterations; ++i) {
        redetermineCentroids();
        if (g_newCentroids[0].isApprox(_centroids)) break;
        else _centroids = g_newCentroids[0];
    }
    std::cout << "Convergence reached on iteration:" << std::endl;
    std::cout << i << std::endl;

    _labels = predict(data);

    return _labels;

}

Eigen::VectorXi metis::KMeans::predict(Eigen::MatrixXd *data) const {

    Eigen::VectorXi labels(data->rows());

    for (unsigned i = 0; i < _nInstances; ++i)
        labels.coeffRef(i) = closestCentroid(data->row(i));

    return labels;

}

Eigen::VectorXi metis::KMeans::predict(metis::DataSet *data) const {
    return predict(data->getData());
}

Eigen::MatrixXd metis::KMeans::getCentroids() const {
    return _centroids;
}

Eigen::VectorXi metis::KMeans::getLabels() const {
    return _labels;
}

void metis::KMeans::setMaxIterations(unsigned iterations) {
    _maxIterations = iterations;
}

double metis::KMeans::score(metis::DataSet *data) const {

    double meanDistance;
    double distance;
    Eigen::MatrixXd *uData = data->getData();

    Eigen::VectorXi labels = predict(data->getData());
    for (unsigned i = 0; i < data->instances(); ++i) {
        distance = (_centroids.row(labels.coeffRef(i)) - uData->row(i)).squaredNorm();
        meanDistance += distance;
    }

    return meanDistance / data->instances();

}

metis::KMeans::KMeans(unsigned nClusters, unsigned initMethod) {
    _nClusters = nClusters;
    _initMethod = initMethod;
}

metis::KMeans::KMeans(unsigned nClusters) {
    _nClusters = nClusters;
}

metis::KMeans::KMeans() {

}

metis::KMeans::~KMeans() {

}
