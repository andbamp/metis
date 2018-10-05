//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MULTINOMIALNBDEMO_H
#define METIS_MULTINOMIALNBDEMO_H

#include "../include/MultinomialNB.h"

void demoMultinomialNB(unsigned nThread) {

    omp_set_num_threads(nThread);

    metis::DataLabeled *bc = metis::loadBC();
    bc->shuffle();
    metis::DataLabeled *bcTest = bc->split(0.1);
//    metis::DataLabeled *bcTest = bc;

    metis::MultinomialNB mnb;

    double before = omp_get_wtime();
    mnb.fit(bc);
    double after = omp_get_wtime();

    Eigen::MatrixXd predictionProb(bcTest->instances(), bcTest->classes(0));
    predictionProb = mnb.findPosterior(bcTest->getInputs()->getData());

    Eigen::MatrixXd comparison(bcTest->instances(), 1+2+bcTest->classes(0)+4+1);
    comparison.col(0) = bcTest->getOutputs()->getData()->col(0);
    comparison.col(2) = mnb.predict(bcTest->getInputs()->getData(), false).cast<double>();
    comparison.col(3) = predictionProb.row(0).transpose();
    comparison.col(4) = predictionProb.row(1).transpose();
    comparison.col(5) = mnb.findEvidence(0, bcTest->getInputs()->getData());
    comparison.col(6) = mnb.findEvidence(1, bcTest->getInputs()->getData());
    comparison.col(7) = mnb.findEvidence(2, bcTest->getInputs()->getData());
    comparison.col(8) = mnb.findEvidence(3, bcTest->getInputs()->getData());
    comparison.col(9) = mnb.findPrior(bcTest->getOutputs()->getData());

    std::cout << "\nPredictions:" << std::endl;
    std::cout << comparison << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << mnb.score(bcTest, false) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}

#endif //METIS_MULTINOMIALNBDEMO_H
