//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/MultinomialNaiveBayes.h"
#include <iostream>

void metis::MultinomialNaiveBayes::fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target,
                                       Eigen::MatrixXi *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) {
    
    determineProbabilities(input, target);
    _nAttributes = input->cols();
    _nClasses = _nConditions;
    
    bool zfp = false;
    
    for (unsigned c = 0; c < _nClasses; ++c) {
        if (_likelihood[c]->zeroFrequencyExists()) {
            zfp = true;
            break;
        }
    }
    
    for (unsigned c = 0; c < _nClasses; ++c) {
        _likelihood[c]->eliminateZeroFrequencies();
    }
    
}

void metis::MultinomialNaiveBayes::fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target, unsigned verboseCycle) {
    Predictor::fit(input, target, verboseCycle);
}

void metis::MultinomialNaiveBayes::fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target,
                                       Eigen::MatrixXi *valInput, Eigen::ArrayXi *valTarget) {
    Predictor::fit(input, target, valInput, valTarget);
}

void metis::MultinomialNaiveBayes::fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target) {
    Predictor::fit(input, target);
}

Eigen::MatrixXd metis::MultinomialNaiveBayes::predictProbabilities(Eigen::MatrixXi *input) const {
    return findPosterior(input);
}

Eigen::MatrixXd metis::MultinomialNaiveBayes::findPosterior(Eigen::MatrixXi *data) const {
    return BayesianModel::findPosterior(data);
}

Eigen::VectorXd metis::MultinomialNaiveBayes::findLikelihood(Eigen::MatrixXi *data, unsigned attr,
                                                             unsigned condition) const {
    return BayesianModel::findLikelihood(data, attr, condition);
}

Eigen::VectorXd metis::MultinomialNaiveBayes::findEvidence(Eigen::MatrixXi *data) const {
    return BayesianModel::findEvidence(data);
}
