//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LOGREGDEMOIRIS_H
#define METIS_LOGREGDEMOIRIS_H

#include "../include/LogisticRegression.h"
#include "../include/ToyDataSets.h"

void demoLogRegIris(unsigned iterations, double learnRate, unsigned batchSize, unsigned nThreads) {

    Eigen::setNbThreads(1);
    omp_set_num_threads(nThreads);

    metis::DataLabeled *iris = metis::loadIris();
    iris->shuffle();
    metis::DataLabeled *irisTest = iris->split(0.25);

    metis::LogisticRegression logreg(iterations, learnRate, batchSize);

    double before = omp_get_wtime();
    logreg.fit(iris);
    double after = omp_get_wtime();

    Eigen::MatrixXd predictionProb(irisTest->instances(), irisTest->classes(0));
    predictionProb = logreg.predictProbabilities(irisTest->getInputs());

    Eigen::MatrixXd comparison(irisTest->instances(), 2+irisTest->classes(0));
    comparison.col(0) = irisTest->getOutputs()->getData()->col(0);
    comparison.col(1) = logreg.predict(irisTest->getInputs()).cast<double>();
    comparison.col(2) = predictionProb.col(0);
    comparison.col(3) = predictionProb.col(1);
    comparison.col(4) = predictionProb.col(2);

    std::cout << "\nPredictions:" << std::endl;
    std::cout << comparison << std::endl;

    std::cout << "\nCoefficients:" << std::endl;
    std::cout << logreg.getCoefficients() << std::endl;

    std::cout << "\nIntercepts:" << std::endl;
    std::cout << logreg.getIntercepts() << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << logreg.score(irisTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}

#endif //METIS_LOGREGDEMOIRIS_H
