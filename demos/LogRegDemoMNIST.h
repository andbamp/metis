//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/ToyDataSets.h"
#include "../include/LogisticRegression.h"

void demoLogRegMNIST(unsigned iterations, double learnRate, unsigned batchSize, unsigned nThreads) {

    Eigen::setNbThreads(1);
    omp_set_num_threads(nThreads);

    metis::DataLabeled *mnist = metis::loadMNIST();
    metis::DataLabeled *mnistTest = metis::loadMNIST(true);

    metis::LogisticRegression logreg(iterations, learnRate, batchSize);

    double before = omp_get_wtime();
    logreg.fit(mnist, mnistTest, false);
    double after = omp_get_wtime();

    Eigen::VectorXi prediction(mnistTest->instances());
    prediction = logreg.predict(mnist->getInputs());

    Eigen::MatrixXd predictionProb(mnistTest->instances(), mnist->classes(0));
    predictionProb = logreg.predictProbabilities(mnistTest->getInputs());

    Eigen::MatrixXd comparison(mnistTest->instances(), 2+mnist->classes(0));
    comparison.col(0) = logreg.predict(mnistTest->getInputs()).cast<double>();
    comparison.col(1) = mnistTest->getOutputs()->getData()->col(0);
    comparison.col(2) = predictionProb.col(0);
    comparison.col(3) = predictionProb.col(1);
    comparison.col(4) = predictionProb.col(2);
    comparison.col(5) = predictionProb.col(3);
    comparison.col(6) = predictionProb.col(4);
    comparison.col(7) = predictionProb.col(5);
    comparison.col(8) = predictionProb.col(6);
    comparison.col(9) = predictionProb.col(7);
    comparison.col(10) = predictionProb.col(8);
    comparison.col(11) = predictionProb.col(9);

//    std::cout << comparison << std::endl;

    std::cout << "\nCoefficients:" << std::endl;
//    std::cout << logreg.getCoefficients() << std::endl;

    std::cout << "\nIntercepts:" << std::endl;
//    std::cout << logreg.getIntercepts() << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << logreg.score(mnistTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}
