//
// Copyright (c) 2018 Andreas Bampouris
//

#include "GaussianDistribution.h"

void metis::GaussianDistribution::fit(Eigen::MatrixXd *data) {

    unsigned nInstances = data->rows();
    unsigned nAttributes = data->cols();
    
    _means.resize(nAttributes);
    _stDev.resize(nAttributes);
    
    for (unsigned a = 0; a < nAttributes; ++a) {
        _means.coeffRef(a) = data->col(a).mean();
        _stDev.coeffRef(a) = std::sqrt( (data->col(a).array() - _means.coeff(a)).square().sum() / (nInstances - 1) );
    }

}

Eigen::MatrixXd metis::GaussianDistribution::findProbability(Eigen::MatrixXd *data) const {
    
    unsigned nInstances = data->rows();
    unsigned nAttributes = data->cols();
    Eigen::MatrixXd probability(nInstances, nAttributes);
    
    for (unsigned a = 0; a < nAttributes; ++a) {
        probability.col(a) = findProbability(data, a);
    }
    
    return probability;
    
}

Eigen::VectorXd metis::GaussianDistribution::findProbability(Eigen::MatrixXd *data, unsigned attr) const {
    
    unsigned nInstances = data->rows();
    Eigen::VectorXd probability(nInstances);
    
    // Applies data to probability density function.
    probability.array() = (data->col(attr).array() - _means.coeff(attr)).square();
    probability.array() /= -2 * pow(_stDev.coeff(attr), 2);
    probability.array() = probability.array().exp();
    probability.array() /= sqrt(2 * M_PI * pow(_stDev.coeff(attr), 2));
    
    return probability;
    
}
