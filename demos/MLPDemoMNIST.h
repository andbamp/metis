//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MLPDEMOMNIST_H
#define METIS_MLPDEMOMNIST_H

#include "../include/ToyDataSets.h"
#include "../include/MLPClassifier.h"

void demoMLPMNIST(std::vector<unsigned> hiddenLayers, std::vector<unsigned> activationFunction,
                  unsigned iterations, double learnRate, unsigned batchSize, unsigned nThreads) {


    std::cout << "\n-------------------------------------" << std::endl;
    omp_set_num_threads(nThreads);

    metis::DataLabeled *mnist = metis::loadMNIST();
    metis::DataLabeled *mnistTest = metis::loadMNIST(true);

    metis::MLPClassifier clf(hiddenLayers, activationFunction, batchSize, learnRate, iterations);

    double before = omp_get_wtime();
    clf.train(mnist, mnistTest, false);
    double after = omp_get_wtime();

    Eigen::VectorXd predictions(mnistTest->instances());
    predictions = clf.predict(mnistTest->getInputs()).cast<double>();

    Eigen::MatrixXd predictionProb(mnistTest->instances(), mnist->classes(0));
    predictionProb = clf.predictProbabilities(mnistTest->getInputs());

//    Eigen::MatrixXd comparison(mnistTest->instances(), 2+mnist->classes(0));
//    comparison.col(0) = mnistTest->getOutputs()->getData()->col(0);
//    comparison.col(1) = predictions;
//    comparison.col(2) = predictionProb.row(0).transpose();
//    comparison.col(3) = predictionProb.row(1).transpose();
//    comparison.col(4) = predictionProb.row(2).transpose();
//    comparison.col(5) = predictionProb.row(3).transpose();
//    comparison.col(6) = predictionProb.row(4).transpose();
//    comparison.col(7) = predictionProb.row(5).transpose();
//    comparison.col(8) = predictionProb.row(6).transpose();
//    comparison.col(9) = predictionProb.row(7).transpose();
//    comparison.col(10) = predictionProb.row(8).transpose();
//    comparison.col(11) = predictionProb.row(9).transpose();
//
//    std::cout << comparison << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << clf.score(mnistTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}


#endif //METIS_MLPDEMOMNIST_H