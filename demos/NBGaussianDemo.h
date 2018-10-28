//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_GAUSSIANNBDEMO_H
#define METIS_GAUSSIANNBDEMO_H

#include "../include/GaussianNB.h"
#include "../include/ToyDataSets.h"

void demoGaussianNB(unsigned nThread) {

    srand(0);
    rand();

    metis::DataLabeled *iris = metis::loadIris();
    iris->shuffle();
    metis::DataLabeled *irisTest = iris->split(0.25);

    metis::GaussianNB gnb;

    double before = omp_get_wtime();
    gnb.fit(iris);
    double after = omp_get_wtime();

    Eigen::MatrixXd predictionProb(irisTest->instances(), irisTest->classes(0));
    predictionProb = gnb.findPosterior(irisTest->getInputs()->getData());

    Eigen::MatrixXd comparison(irisTest->instances(), 1+2+irisTest->classes(0));
    comparison.col(0) = irisTest->getOutputs()->getData()->col(0);
    comparison.col(2) = gnb.predict(irisTest->getInputs()->getData(), false).cast<double>();
    comparison.col(3) = predictionProb.row(0).transpose();
    comparison.col(4) = predictionProb.row(1).transpose();
    comparison.col(5) = predictionProb.row(2).transpose();

    std::cout << "\nPredictions:" << std::endl;
//    std::cout << comparison << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << gnb.score(irisTest, false) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}

#endif //METIS_GAUSSIANNBDEMO_H
