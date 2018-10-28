//
// Copyright (c) 2018 Andreas Bampouris
//

#include "BayesianModel.h"
#include "GaussianDistribution.h"
#include "../../include/DataContainer.h"
#include <iostream>

template<class D, class P>
void metis::BayesianModel<D, P>::determineProbabilities(D *input, Eigen::ArrayXi *target) {
    
    //!!! Extra cost a result of separating loop that took care of both prior and likelihood
    double before = omp_get_wtime();
    
    _prior.fit(target);
    _evidence.fit(input);
    
    double after = omp_get_wtime();
    std::cout << "1: " << after - before << "sec" << std::endl;
    
    //!!! Costliest operation here
    before = omp_get_wtime();
    std::vector<D *> dividedData = DataContainer::createPerClassMatrices<D>(input, target);
    after = omp_get_wtime();
    std::cout << "2: " << after - before << "sec" << std::endl;
    
    //!!! In other implementation, only indices of separated data are kept, and datasets are created on-demand
    // Never are all divided datasets kept in memory at the same time
    before = omp_get_wtime();
    _nConditions = dividedData.size();
    _likelihood.resize(_nConditions);
    for (unsigned c = 0; c < _nConditions; ++c) {
        
        _likelihood[c] = new P();
        _likelihood[c]->fit(dividedData[c]);
        delete dividedData[c];
        
    }
    after = omp_get_wtime();
    std::cout << "3: " << after - before << "sec" << std::endl;
    
}

template<class D, class P>
Eigen::MatrixXd metis::BayesianModel<D, P>::findPosterior(D *data) const {
    
    unsigned nInstances = data->rows();
    unsigned nAttributes = data->cols();
    
    Eigen::MatrixXd probabilities(nInstances, _nConditions);
    probabilities.setOnes();
    
    for (unsigned c = 0; c < _nConditions; ++c) {
        for (unsigned a = 0; a < nAttributes; ++a) {
            probabilities.array().col(c) *= findLikelihood(data, a, c).array();
        }
        probabilities.array().col(c) *= findPrior(c).coeff(0);
    }
    
    return probabilities;
    
}

template<class D, class P>
Eigen::VectorXd metis::BayesianModel<D, P>::findPrior(Eigen::ArrayXi *data) const {
    Eigen::MatrixXi mData(data->rows(), 1);
    mData.col(0).array() = *data;
    return _prior.findProbability(&mData);
}

template<class D, class P>
Eigen::VectorXd metis::BayesianModel<D, P>::findPrior(unsigned condition) const {
    Eigen::VectorXd priorProb(1);
    priorProb.coeffRef(0) = _prior.getFrequency(condition);
    return priorProb;
}

template<class D, class P>
Eigen::VectorXd metis::BayesianModel<D, P>::findLikelihood(D *data, unsigned attr, unsigned condition) const {
    return _likelihood[condition]->findProbability(data, attr);
}

template<class D, class P>
Eigen::VectorXd metis::BayesianModel<D, P>::findEvidence(D *data) const {
    return _evidence.findProbability(data);
}

template class metis::BayesianModel<Eigen::MatrixXi, metis::Frequency>;
template class metis::BayesianModel<Eigen::MatrixXd, metis::GaussianDistribution>;
