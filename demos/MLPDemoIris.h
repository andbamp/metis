//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MLPDEMOIRIS_H
#define METIS_MLPDEMOIRIS_H

#include "../include/ToyDataSets.h"
#include "../include/MLPClassifier.h"

void demoMLPIris(std::vector<unsigned> hiddenLayers, std::vector<unsigned> activationFunction,
                 unsigned batchSize, double learnRate, unsigned iterations, unsigned nThreads) {

    omp_set_num_threads(nThreads);

    metis::DataLabeled *iris = metis::loadIris();
    iris->shuffle();
    metis::DataLabeled *irisTest = iris->split(0.3);

    metis::MLPClassifier clf(hiddenLayers, activationFunction, batchSize, learnRate, iterations);

    double before = omp_get_wtime();
    clf.train(iris, irisTest);
    double after = omp_get_wtime();

    Eigen::MatrixXd predictionProb(irisTest->instances(), iris->classes(0));
    predictionProb = clf.predictProbabilities(irisTest->getInputs());

    Eigen::MatrixXd comparison(irisTest->instances(), 2+iris->classes(0));
    comparison.col(0) = irisTest->getOutputs()->getData()->col(0);
    comparison.col(1) = clf.predict(irisTest->getInputs()).cast<double>();
    comparison.col(2) = predictionProb.row(0).transpose();
    comparison.col(3) = predictionProb.row(1).transpose();
    comparison.col(4) = predictionProb.row(2).transpose();

    std::cout << comparison << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << clf.score(irisTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}


#endif //METIS_MLPDEMOIRIS_H