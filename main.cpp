//
// Copyright (c) 2018 Andreas Bampouris
//

#include <iostream>
#include <omp.h>
#include "demos/MetisDemos.h"
#include "include/DataContainer.h"
#include "src/core/GaussianDistribution.h"

//#define EIGEN_DONT_PARALLELIZE

void demoTest();

int main() {

    srand(time(NULL));
    rand();
    
//    demoTest();
    
//    demoDiabetesLinearRegression(true, 800000, 0.01, 64, 1, 2, false);
//    demoDiabetesLinearRegression(false, 2, 1, false);
//
//    demoBanknotesLogisticRegression(3000, 0.1, 64, 1, 2, false);
//    demoIrisLogisticRegression(3000000, 0.1, 64, 1, 2, false);
//    demoMNISTLogisticRegression(10, 0.1, 64, 1, 1, false);
//    demoMNISTLogisticRegression(10, 0.1, 64, 1, 2, false);


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

void demoTest() {
    
    metis::DataContainer *iris = metis::loadIrisContainer();
    Eigen::MatrixXd input = iris->createNumericalMatrix({0,1,2,3});
    Eigen::ArrayXi target = iris->createClassArray(4);
    
    std::vector<metis::GaussianDistribution *> distros = metis::GaussianDistribution::createClassDistributions(&input, &target);
    std::cout << distros[0]->findProbability(&input) << std::endl << std::endl;
    std::cout << distros[1]->findProbability(&input) << std::endl << std::endl;
    std::cout << distros[2]->findProbability(&input) << std::endl << std::endl;
    
}
