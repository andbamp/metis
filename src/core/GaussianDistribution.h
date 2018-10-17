//
// Created by Andreas Bampouris on 14/10/2018.
//

#ifndef METIS_GAUSSIANDISTRIBUTION_H
#define METIS_GAUSSIANDISTRIBUTION_H

#include <Eigen/Dense>
#include <vector>

namespace metis {

class GaussianDistribution {

private:

    // Data
    Eigen::VectorXd _means;
    Eigen::VectorXd _stDev;
    
    // Meta-data
    unsigned _nAttributes = 1;

public:

    // Access
    /**
     * Creates a gaussian distribution based on a given data set.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance.
     */
    void fit(Eigen::MatrixXd *data);
    
    /**
     * Calculates probabilities for input data of one attribute based on normal distribution.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance.
     * @param attr Attribute for which probability is calculated.
     * @return VectorXd with probabilities.
     */
    Eigen::VectorXd findProbability(Eigen::MatrixXd *data, unsigned attr) const;
    
    /**
     * Calculates probabilities for input data based on normal distribution.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance.
     * @return MatrixXd with probabilities.
     */
    Eigen::MatrixXd findProbability(Eigen::MatrixXd *data) const;
    
    /**
     * Creates a gaussian distributions for data of each class.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance.
     * @param target Data set represented as an ArrayXi. Each row represents an instance. Same size as data.
     * @return C++ vector of as many pointers to GaussianDistribution objects as there are classes.
     */
    static std::vector<GaussianDistribution *> createClassDistributions(Eigen::MatrixXd *data, Eigen::ArrayXi *target);
    
    // Construction
    GaussianDistribution();
    ~GaussianDistribution();
    
};

}


#endif //METIS_GAUSSIANDISTRIBUTION_H
