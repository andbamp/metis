//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Frequency.h"
#include "../../include/DataContainer.h"

void metis::Frequency::fit(Eigen::MatrixXi *data) {

    unsigned nInstances = data->rows();
    unsigned nAttributes = data->cols();
    
    // Finds number of categories for each attribute.
    Eigen::ArrayXi nCategories = DataContainer::findNumberOfCategories(data);
    _frequencies.resize(nAttributes);
    
    // Goes through data to find frequencies of each distinct value of each attribute.
    for (unsigned a = 0; a < nAttributes; ++a) {
    
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
    unsigned nAttributes = data->cols();
    Eigen::MatrixXd probability(nInstances, nAttributes);
    
    for (unsigned a = 0; a < nAttributes; ++a) {
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

double metis::Frequency::getFrequency(unsigned value, unsigned attribute) {
    return _frequencies[attribute].coeff(value);
}

double metis::Frequency::getFrequency(unsigned value) {
    // When no attribute is given, index 0 of std::vector is assumed.
    return getFrequency(value, 0);
}
