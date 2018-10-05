//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LOGREGDEMOBANKNOTES_H
#define METIS_LOGREGDEMOBANKNOTES_H

#include "../include/LogisticRegression.h"
#include "../include/ToyDataSets.h"

void demoLogRegBanknotes(unsigned iterations, double learnRate, unsigned batchSize, unsigned nThreads) {

    omp_set_num_threads(nThreads);

    metis::DataLabeled *notes = metis::loadBanknotes();
    notes->shuffle();
    metis::DataLabeled *notesTest = notes->split(0.3);

    metis::LogisticRegression logreg(iterations, learnRate, batchSize);

    double before = omp_get_wtime();
    logreg.fit(notes, notesTest, true);
    double after = omp_get_wtime();

    Eigen::VectorXi prediction(notesTest->instances());
    prediction = logreg.predict(notesTest->getInputs());

    Eigen::MatrixXd predictionProb(notesTest->instances(), notesTest->classes(0));
    predictionProb = logreg.predictProbabilities(notesTest->getInputs());

    Eigen::MatrixXd comparison(notesTest->instances(), 2+notesTest->classes(0));
    comparison.col(0) = notesTest->getOutputs()->getData()->col(0);
    comparison.col(1) = logreg.predict(notesTest->getInputs()).cast<double>();
    comparison.col(2) = predictionProb.col(0);
    comparison.col(3) = predictionProb.col(1);

    std::cout << comparison << std::endl;

    std::cout << "\nCoefficients:" << std::endl;
    std::cout << logreg.getCoefficients() << std::endl;

    std::cout << "\nIntercepts:" << std::endl;
    std::cout << logreg.getIntercepts() << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << logreg.score(notesTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}

#endif //METIS_LOGREGDEMOBANKNOTES_H
