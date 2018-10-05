//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/DecisionTree.h"

void demoDTree(unsigned nThreads) {

    omp_set_num_threads(nThreads);

    // Creating data set
    metis::DataLabeled *bc = metis::loadBC();
    bc->shuffle();
//    metis::DataLabeled *bcTest = bc->split(0.1);
    metis::DataLabeled *bcTest = bc;

    // Fitting
    double before = omp_get_wtime();
    metis::DecisionTree tree;
    tree.fit(bc);
    double after = omp_get_wtime();

    // Results
    Eigen::MatrixXd comparison(bcTest->instances(), 2);
    comparison.col(0) = bcTest->getOutputs()->getData()->col(0);
    comparison.col(1) = tree.predict(bcTest->getInputs()).cast<double>();

    std::cout << "\nTree:" << std::endl;
    tree.print();

    std::cout << "\nPredictions:" << std::endl;
    std::cout << comparison << std::endl;

    std::cout << "\nAccuracy on test set:" << std::endl;
    std::cout << tree.score(bcTest) << std::endl;

    std::cout << "\nComputation of clusters took:" << std::endl;
    std::cout << (after - before) << " sec" << std::endl;

}
