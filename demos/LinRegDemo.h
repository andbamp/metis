//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/ToyDataSets.h"
#include "../include/LinearRegression.h"

void demoLinReg(unsigned nThreads) {

    Eigen::setNbThreads(1);
    omp_set_num_threads(nThreads);

    metis::DataLabeled *diabetes = metis::loadDiabetes();
    diabetes->shuffle();
    metis::DataLabeled *diabetesTest = diabetes->split(0.2);

    metis::LinearRegression regressor = metis::LinearRegression();

    double before = omp_get_wtime();
    regressor.fit(diabetes);
    double after = omp_get_wtime();

    std::cout << "\nModel has been fitted:" << std::endl;
    std::cout << regressor.getCoefficients() << std::endl;
    std::cout << std::endl << regressor.getIntercept() << std::endl;

    Eigen::MatrixXd prediction(diabetesTest->instances(), diabetesTest->outputs());
    prediction = regressor.predict(diabetesTest->getInputs()->getData());

    Eigen::MatrixXd answer(diabetesTest->instances(), 2 * diabetesTest->outputs());
    answer.col(0) = diabetesTest->getOutputs()->getData()->col(0);
    answer.col(1) = diabetesTest->getOutputs()->getData()->col(1);
    answer.col(2) = prediction.col(0);
    answer.col(3) = prediction.col(1);

    std::cout << "\nPredictions and targets:" << std::endl;
    std::cout << answer << std::endl;

    std::cout << "\nMean squared error:" << std::endl;
    std::cout << regressor.score(diabetesTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;

}
