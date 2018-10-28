//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/ToyDataSets.h"
#include "../include/GaussianNaiveBayes.h"
#include "../include/MultinomialNaiveBayes.h"

void demoBCMultinomialNaiveBayes(unsigned eigThreads, unsigned ompThreads, bool verbose) {
    
    if (ompThreads > 1) {
#define EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(1);
        omp_set_num_threads(ompThreads);
    } else {
        Eigen::setNbThreads(eigThreads);
        omp_set_num_threads(1);
    }
    
    // Loading data containers.
    metis::DataContainer *bc = metis::loadBCContainer();
    bc->shuffle();
    metis::DataContainer bcTest = bc->split(0.2);
//    metis::DataContainer bcTest = *bc;
    
    // Creating train set.
    Eigen::MatrixXi input = bc->createCategoricalMatrix({1,2,3,4,5,6,7,8,9});
    Eigen::ArrayXi target = bc->createClassArray(0);
    
    // Creating test set.
    Eigen::MatrixXi inputTest = bcTest.createCategoricalMatrix({1,2,3,4,5,6,7,8,9});
    Eigen::ArrayXi targetTest = bcTest.createClassArray(0);
    
    double before = omp_get_wtime();
    
    // Training.
    metis::MultinomialNaiveBayes classifier;
    classifier.fit(&input, &target);
    
    double after = omp_get_wtime();
    
    if (verbose) {
        
        // Printing parameters.
//        std::cout << "\nModel has been fitted:" << std::endl;
//        std::cout << classifier.getPrior() << std::endl;
//        std::cout << std::endl << classifier.getLikelihood() << std::endl;
//        std::cout << std::endl << classifier.getEvidence() << std::endl;
        
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

void demoBCMultinomialNaiveBayes() {
    demoBCMultinomialNaiveBayes(1, 1, true);
}

void demoIrisGaussianNaiveBayes(unsigned eigThreads, unsigned ompThreads, bool verbose) {
    
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
    metis::GaussianNaiveBayes classifier;
    classifier.fit(&input, &target);
    
    double after = omp_get_wtime();
    
    if (verbose) {
    
        // Printing parameters.
//        std::cout << "\nModel has been fitted:" << std::endl;
//        std::cout << classifier.getPrior() << std::endl;
//        std::cout << std::endl << classifier.getLikelihood() << std::endl;
//        std::cout << std::endl << classifier.getEvidence() << std::endl;
    
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

void demoIrisGaussianNaiveBayes() {
    demoIrisGaussianNaiveBayes(1, 1, true);
}
