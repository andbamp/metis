//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_GAUSSIANDISTRIBUTION_H
#define METIS_GAUSSIANDISTRIBUTION_H


#include "Probability.h"

namespace metis {

class GaussianDistribution : public Probability<Eigen::MatrixXd> {

private:

    // Data
    
    //! Mean for each attribute.
    Eigen::VectorXd _means;
    
    //! Standard deviation for each attribute.
    Eigen::VectorXd _stDev;

public:

    // Access
    
    /**
     * Calculates mean and standard deviation for each attribute of the data.
     *
     * @param data Data for which the probabilistic model is built.
     */
    void fit(Eigen::MatrixXd *data) override;
    
    /**
     * Calculates probabilities of input data assuming normal distribution.
     *
     * @param data Data the probabilities of which are calculated.
     * @return MatrixXd for probabilities of values of each attribute.
     */
    Eigen::MatrixXd findProbability(Eigen::MatrixXd *data) const override;
    Eigen::VectorXd findProbability(Eigen::MatrixXd *data, unsigned attr) const override;
    
};

}


#endif //METIS_GAUSSIANDISTRIBUTION_H
