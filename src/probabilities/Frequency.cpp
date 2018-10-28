//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Frequency.h"
#include "../../include/DataContainer.h"

template <class I>
void metis::Frequency::fit(I *data) {

    _nInstances = data->rows();
    unsigned nAttributes = data->cols();
    
    // Finds number of categories for each attribute.
    _nCategories = DataContainer::findNumberOfCategories(data);
    _frequencies.resize(nAttributes);
    
    // Goes through data to find frequencies of each distinct value of each attribute.
    for (unsigned a = 0; a < nAttributes; ++a) {
        
        _frequencies[a].resize(_nCategories.coeff(a));
        _frequencies[a].setZero();
        
        for (unsigned i = 0; i < _nInstances; ++i) {
            _frequencies[a].coeffRef(data->coeff(i, a))++;
        }
        
        _frequencies[a] /= _nInstances;
        
    }

}

void metis::Frequency::fit(Eigen::MatrixXi *data) {
    fit<Eigen::MatrixXi>(data);
}

void metis::Frequency::fit(Eigen::VectorXi *data) {
    fit<Eigen::VectorXi>(data);
}

void metis::Frequency::fit(Eigen::ArrayXi *data) {
    fit<Eigen::ArrayXi>(data);
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

double metis::Frequency::getFrequency(unsigned value, unsigned attribute) const {
    return _frequencies[attribute].coeff(value);
}

double metis::Frequency::getFrequency(unsigned value) const {
    // When no attribute is given, index 0 of std::vector is assumed.
    return getFrequency(value, 0);
}

void metis::Frequency::eliminateZeroFrequencies() {
    
    unsigned nAttributes = _frequencies.size();
    
    // Solves the zero-frequency problem by adding 1 to the frequency of every value of every attribute.
    for (unsigned a = 0; a < nAttributes; ++a) {
        _frequencies[a] *= _nInstances;
        _frequencies[a].array() += 1;
        _frequencies[a] /= _nInstances + _nCategories.coeff(a);
    }
    
}

bool metis::Frequency::zeroFrequencyExists() {
    
    unsigned nAttributes = _frequencies.size();
    
    for (unsigned a = 0; a < nAttributes; ++a) {
        for (unsigned c = 0; c < _nCategories[a]; ++c) {
            if (_frequencies[a].coeff(c) == 0) {
                return true;
            }
        }
    }
    
    return false;
    
}
