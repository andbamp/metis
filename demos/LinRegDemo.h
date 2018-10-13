//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/ToyDataSets.h"
#include "../include/LinearRegression.h"

void demoLinReg(bool iterative, unsigned iterations, double learnRate, unsigned batchSize,
        unsigned eigThreads, unsigned ompThreads, bool verbose) {

    if (ompThreads > 1) {
#define EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(1);
        omp_set_num_threads(ompThreads);
    } else {
        Eigen::setNbThreads(eigThreads);
        omp_set_num_threads(1);
    }
    
    // Loading data containers.
    metis::DataContainer *diabetes = metis::loadDiabetesContainer();
    diabetes->shuffle();
    metis::DataContainer diabetesTest = diabetes->split(0.25);
    
    // Creating train set.
    Eigen::MatrixXd input = diabetes->createNumericalMatrix({0,1,2,3,4,5,6,7,8,9});
    Eigen::MatrixXd target = diabetes->createNumericalMatrix({10,11});
    
    // Creating test set.
    Eigen::MatrixXd inputTest = diabetesTest.createNumericalMatrix({0,1,2,3,4,5,6,7,8,9});
    Eigen::MatrixXd targetTest = diabetesTest.createNumericalMatrix({10,11});
    
    double before = omp_get_wtime();
    
    // Training.
    metis::LinearRegression regressor(iterative, iterations, learnRate, batchSize);
    regressor.fit(&input, &target);
    
    double after = omp_get_wtime();

    if (verbose) {
        // Printing parameters.
        std::cout << "\nModel has been fitted:" << std::endl;
        std::cout << regressor.getCoefficients() << std::endl;
        std::cout << std::endl << regressor.getIntercepts() << std::endl;
    
        // Making predictions.
        Eigen::MatrixXd prediction(targetTest.rows(), targetTest.cols());
        prediction = regressor.predict(&inputTest);
    
        // Comparing predictions and targets.
        Eigen::MatrixXd comparison(targetTest.rows(), 2 * targetTest.cols());
        comparison.leftCols(2) = targetTest;
        comparison.rightCols(2) = prediction;
    
        std::cout << "\nTargets and predictions:" << std::endl;
        std::cout << comparison << std::endl;
    }

    std::cout << "\nMean squared error:" << std::endl;
    std::cout << regressor.score(&inputTest, &targetTest) << std::endl;

    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;
    
    std::cout << "\n\t---------------------\n" << std::endl;
    
    if (ompThreads > 1) {
#undef EIGEN_DONT_PARALLELIZE
    }

}

void demoLinReg(bool iterative, unsigned eigThreads, unsigned ompThreads, bool verbose) {
    demoLinReg(iterative, 10, 0.01, 1, eigThreads, ompThreads, verbose);
}