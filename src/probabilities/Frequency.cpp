//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Frequency.h"
#include "../../include/DataContainer.h"

void metis::Frequency::fit(Eigen::MatrixXi *data) {

    _nAttributes = data->cols();
    unsigned nInstances = data->rows();
    
    // Finds number of categories for each attribute.
    Eigen::ArrayXi nCategories = DataContainer::findNumberOfCategories(data);
    _frequencies.resize(_nAttributes);
    
    // Goes through data to find frequencies of each distinct value of each attribute.
    for (unsigned a = 0; a < _nAttributes; ++a) {
    
        _frequencies[a].resize(nCategories.coeff(a));
        _frequencies[a].setZero();
        
        for (unsigned i = 0; i < nInstances; ++i) {
            _frequencies[a].coeffRef(data->coeff(i, a))++;
        }
        
        _frequencies[a] /= nInstances;
        
    }

}

Eigen::MatrixXd metis::Frequency::findProbability(Eigen::MatrixXi *data) const {
    
    unsigned nInstances = data->rows();
    Eigen::MatrixXd probability(nInstances, _nAttributes);
    
    for (unsigned a = 0; a < _nAttributes; ++a) {
        for (unsigned i = 0; i < nInstances; ++i) {
            probability.coeffRef(i, a) = _frequencies[a].coeff(data->coeff(a));
        }
    }
    
    return probability;
    
}

Eigen::VectorXd metis::Frequency::findProbability(Eigen::MatrixXi *data, unsigned attr) const {
    
    unsigned nInstances = data->rows();
    Eigen::VectorXd probability(nInstances);
    
    for (unsigned i = 0; i < nInstances; ++i) {
        probability.coeffRef(i) = _frequencies[attr].coeff(data->coeff(i, attr));
    }
    
    return probability;
    
}
