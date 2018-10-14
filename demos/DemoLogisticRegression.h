//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LOGREGDEMOIRIS_H
#define METIS_LOGREGDEMOIRIS_H

#include "../include/LogisticRegression.h"
#include "../include/ToyDataSets.h"

void demoBanknotesLogisticRegression(unsigned iterations, double learnRate, unsigned batchSize,
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
    metis::DataContainer *banknotes = metis::loadBanknotesContainer();
    banknotes->shuffle();
    metis::DataContainer banknotesTest = banknotes->split(0.25);
    
    // Creating train set.
    Eigen::MatrixXd input = banknotes->createNumericalMatrix({0,1,2,3});
    Eigen::ArrayXi target = banknotes->createClassArray(4);
    
    // Creating test set.
    Eigen::MatrixXd inputTest = banknotesTest.createNumericalMatrix({0,1,2,3});
    Eigen::ArrayXi targetTest = banknotesTest.createClassArray(4);
    
    double before = omp_get_wtime();
    
    // Training.
    metis::LogisticRegression classifier(iterations, learnRate, batchSize);
    classifier.fit(&input, &target);
    
    double after = omp_get_wtime();
    
    if (verbose) {
        
        // Printing parameters.
        std::cout << "\nModel has been fitted:" << std::endl;
        std::cout << classifier.getCoefficients() << std::endl;
        std::cout << std::endl << classifier.getIntercepts() << std::endl;
        
        // Making predictions.
        Eigen::ArrayXi prediction(targetTest.rows());
        prediction = classifier.predict(&inputTest);
        
        // Comparing predictions and targets.
        Eigen::ArrayXXi comparison(targetTest.rows(), 2 * targetTest.cols());
        comparison.leftCols(targetTest.cols()) = targetTest;
        comparison.rightCols(targetTest.cols()) = prediction;
        
        std::cout << "\nTargets and predictions:" << std::endl;
        std::cout << comparison << std::endl;
        
    }
    
    std::cout << "\nAccuracy:" << std::endl;
    std::cout << classifier.score(&inputTest, &targetTest) << std::endl;
    
    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;
    
    std::cout << "\n\t---------------------\n" << std::endl;
    
    if (ompThreads > 1) {
#undef EIGEN_DONT_PARALLELIZE
    }
    
}

void demoIrisLogisticRegression(unsigned iterations, double learnRate, unsigned batchSize,
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
    metis::DataContainer *iris = metis::loadIrisContainer();
    iris->shuffle();
    metis::DataContainer irisTest = iris->split(0.25);
    
    // Creating train set.
    Eigen::MatrixXd input = iris->createNumericalMatrix({0,1,2,3});
    Eigen::ArrayXi target = iris->createClassArray(4);
    
    // Creating test set.
    Eigen::MatrixXd inputTest = irisTest.createNumericalMatrix({0,1,2,3});
    Eigen::ArrayXi targetTest = irisTest.createClassArray(4);
    
    double before = omp_get_wtime();
    
    // Training.
    metis::LogisticRegression classifier(iterations, learnRate, batchSize);
    classifier.fit(&input, &target);
    
    double after = omp_get_wtime();
    
    if (verbose) {
    
        // Printing parameters.
        std::cout << "\nModel has been fitted:" << std::endl;
        std::cout << classifier.getCoefficients() << std::endl;
        std::cout << std::endl << classifier.getIntercepts() << std::endl;
    
        // Making predictions.
        Eigen::ArrayXi prediction(targetTest.rows());
        prediction = classifier.predict(&inputTest);
    
        // Comparing predictions and targets.
        Eigen::ArrayXXi comparison(targetTest.rows(), 2 * targetTest.cols());
        comparison.leftCols(targetTest.cols()) = targetTest;
        comparison.rightCols(targetTest.cols()) = prediction;
    
        std::cout << "\nTargets and predictions:" << std::endl;
        std::cout << comparison << std::endl;
    
    }
    
    std::cout << "\nAccuracy:" << std::endl;
    std::cout << classifier.score(&inputTest, &targetTest) << std::endl;
    
    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;
    
    std::cout << "\n\t---------------------\n" << std::endl;
    
    if (ompThreads > 1) {
#undef EIGEN_DONT_PARALLELIZE
    }
    
}

void demoMNISTLogisticRegression(unsigned iterations, double learnRate, unsigned batchSize,
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
    metis::DataContainer *mnist = metis::loadMNISTContainer();
    metis::DataContainer *mnistTest = metis::loadMNISTContainer(true);
    
    std::vector<unsigned> attr;
    for (unsigned a = 1; a < 785; ++a) attr.push_back(a);
    
    // Creating train set.
    Eigen::MatrixXd input = mnist->createNumericalMatrix(attr);
    Eigen::ArrayXi target = mnist->createClassArray(0);
    
    // Creating test set.
    Eigen::MatrixXd inputTest = mnistTest->createNumericalMatrix(attr);
    Eigen::ArrayXi targetTest = mnistTest->createClassArray(0);
    
    double before = omp_get_wtime();
    
    // Training.
    metis::LogisticRegression classifier(iterations, learnRate, batchSize);
    classifier.fit(&input, &target);
    
    double after = omp_get_wtime();
    
    if (verbose) {
        
        // Printing parameters.
        std::cout << "\nModel has been fitted:" << std::endl;
        std::cout << classifier.getCoefficients() << std::endl;
        std::cout << std::endl << classifier.getIntercepts() << std::endl;
        
        // Making predictions.
        Eigen::ArrayXi prediction(targetTest.rows());
        prediction = classifier.predict(&inputTest);
        
        // Comparing predictions and targets.
        Eigen::ArrayXXi comparison(targetTest.rows(), 2 * targetTest.cols());
        comparison.leftCols(targetTest.cols()) = targetTest;
        comparison.rightCols(targetTest.cols()) = prediction;
        
        std::cout << "\nTargets and predictions:" << std::endl;
        std::cout << comparison << std::endl;
        
    }
    
    std::cout << "\nAccuracy:" << std::endl;
    std::cout << classifier.score(&inputTest, &targetTest) << std::endl;
    
    std::cout << "\nComputation took:" << std::endl;
    std::cout << after - before << "sec" << std::endl;
    
    std::cout << "\n\t---------------------\n" << std::endl;
    
    if (ompThreads > 1) {
#undef EIGEN_DONT_PARALLELIZE
    }
    
}

#endif //METIS_LOGREGDEMOIRIS_H
