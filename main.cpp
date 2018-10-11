//
// Copyright (c) 2018 Andreas Bampouris
//

#include <iostream>
#include <omp.h>
#include "demos/DemosMetis.h"
#include "include/DataContainer.h"
#include "include/LinearRegressionNew.h"

#define EIGEN_DONT_PARALLELIZE

void demoNew();

int main() {

    srand(time(NULL));
    rand();
    
    demoNew();

    demoLinReg(1);

//    demoLogRegIris(10, 0.1, 1, 1);
//    demoLogRegIris(100, 0.1, 1, 1);
//    demoLogRegIris(1000, 0.1, 1, 1);

//    demoLogRegMNIST(1, 0.01, 60, 1);
//    demoLogRegMNIST(1, 0.01, 60, 2);

//    demoMLPMNIST({100}, {0,0}, 1, 0.01, 60, 2);
//    demoMLPMNIST({100}, {1,0}, 1, 0.01, 60, 2);
//    demoMLPMNIST({1000}, {1,0}, 1, 0.01, 60, 2);
//    demoMLPMNIST({1000,400}, {1,1,0}, 1, 0.01, 60, 2);
//    demoMLPMNIST({1000,400}, {1,1,0}, 1, 0.01, 60, 1);

//    demoGaussianNB(1);
//    demoMultinomialNB(1);

//    demoDTree(1);

//    demoKMeansIris(1, 1);
//    demoKMeansIris(2, 1);
//    demoKMeansIris(3, 1);
//    demoKMeansIris(4, 1);
//    demoKMeansIris(5, 1);
//    demoKMeansIris(6, 1);
//    demoKMeansIris(7, 1);
//    demoKMeansIris(8, 1);
//    demoKMeansIris(9, 1);
//    demoKMeansIris(10, 1);
//
//    demoKMeansIris(3, 0);

}

void demoNew() {
    
    std::string filePath = "../data/diabetes.tab.txt";
    std::string missingValue = "?";
    metis::DataContainer diabetes = metis::DataContainer(filePath, '\t', missingValue, {0,1,2,3,4,5,6,7,8,9,10,11}, {});
    diabetes.standardize({0,1,2,3,4,5,6,7,8,9});
    diabetes.rescale({0,1,2,3,4,5,6,7,8,9});
    diabetes.shuffle();
    metis::DataContainer diabetesTest = diabetes.split(0.25);
    
    Eigen::MatrixXd input = diabetes.createNumericalMatrix({0,1,2,3,4,5,6,7,8,9});
    Eigen::MatrixXd target = diabetes.createNumericalMatrix({10,11});
    
    Eigen::MatrixXd inputTest = diabetesTest.createNumericalMatrix({0,1,2,3,4,5,6,7,8,9});
    Eigen::MatrixXd targetTest = diabetesTest.createNumericalMatrix({10,11});
    
    metis::LinearRegressionNew linreg(true, 100000, 0.01, 1);
    linreg.fit(&input, &target);
    
    Eigen::MatrixXd prediction(inputTest.rows(), 2);
    prediction = linreg.predict(&inputTest);
    
    Eigen::MatrixXd comparison(inputTest.rows(), 4);
    comparison.leftCols(2) = targetTest;
    comparison.rightCols(2) = prediction;
    std::cout << comparison << std::endl;
    
}
