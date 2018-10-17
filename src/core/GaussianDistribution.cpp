//
// Created by Andreas Bampouris on 14/10/2018.
//

#include "GaussianDistribution.h"
#include "../../include/DataContainer.h"
#include <cmath>

void metis::GaussianDistribution::fit(Eigen::MatrixXd *data) {

    unsigned nInstances = data->rows();
    _nAttributes = data->cols();
    
    _means.resize(_nAttributes);
    _stDev.resize(_nAttributes);
    
    for (unsigned a = 0; a < _nAttributes; ++a) {
        _means.coeffRef(a) = data->col(a).mean();
        _stDev.coeffRef(a) = std::sqrt( (data->col(a).array() - _means.coeff(a)).square().sum() / (nInstances - 1) );
    }

}

Eigen::VectorXd metis::GaussianDistribution::findProbability(Eigen::MatrixXd *data, unsigned attr) const {
    
    Eigen::VectorXd likelihoods(data->rows());
    
    likelihoods.array() = (data->col(attr).array() - _means.coeff(attr)).square();
    likelihoods.array() /= -2 * pow(_stDev.coeff(attr), 2);
    likelihoods.array() = likelihoods.array().exp();
    likelihoods.array() /= sqrt(2 * M_PI * pow(_stDev.coeff(attr), 2));
    
    return likelihoods;
    
}

Eigen::MatrixXd metis::GaussianDistribution::findProbability(Eigen::MatrixXd *data) const {
    
    Eigen::MatrixXd likelihoods(data->rows(), _nAttributes);
    
    for (unsigned a = 0; a < _nAttributes; ++a) {
        likelihoods.col(a) = findProbability(data, a);
    }
    
    return likelihoods;
    
}

std::vector<metis::GaussianDistribution *> metis::GaussianDistribution::createClassDistributions(
                                           Eigen::MatrixXd *data, Eigen::ArrayXi *target) {
    
    std::vector<Eigen::MatrixXd *> dividedData = DataContainer::createPerClassMatrices(data, target);
    unsigned nClasses = dividedData.size();
    
    // A number of distributions equal to the number of classes is created.
    std::vector<metis::GaussianDistribution *> distributions(nClasses);
    for (unsigned c = 0; c < nClasses; ++c) {
        
        distributions[c] = new GaussianDistribution();
        distributions[c]->fit(dividedData[c]);
        
    }
    
    return distributions;
    
}

metis::GaussianDistribution::GaussianDistribution() {

}

metis::GaussianDistribution::~GaussianDistribution() {

}
