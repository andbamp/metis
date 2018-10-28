//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/GaussianNaiveBayes.h"

void metis::GaussianNaiveBayes::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
                                    Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) {
    
    determineProbabilities(input, target);
    _nAttributes = input->cols();
    _nClasses = _nConditions;
    
}

void metis::GaussianNaiveBayes::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target, unsigned verboseCycle) {
    Predictor::fit(input, target, verboseCycle);
}

void metis::GaussianNaiveBayes::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target, Eigen::MatrixXd *valInput,
                                    Eigen::ArrayXi *valTarget) {
    Predictor::fit(input, target, valInput, valTarget);
}

void metis::GaussianNaiveBayes::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target) {
    Predictor::fit(input, target);
}

Eigen::MatrixXd metis::GaussianNaiveBayes::predictProbabilities(Eigen::MatrixXd *input) const {
    return findPosterior(input);
}

Eigen::MatrixXd metis::GaussianNaiveBayes::findPosterior(Eigen::MatrixXd *data) const {
    return BayesianModel::findPosterior(data);
}

Eigen::VectorXd metis::GaussianNaiveBayes::findLikelihood(Eigen::MatrixXd *data, unsigned attr,
                                                          unsigned condition) const {
    return BayesianModel::findLikelihood(data, attr, condition);
}

Eigen::VectorXd metis::GaussianNaiveBayes::findEvidence(Eigen::MatrixXd *data) const {
    return BayesianModel::findEvidence(data);
}
