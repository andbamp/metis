//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/MultinomialNB.h"

void metis::MultinomialNB::fit(metis::DataLabeled *data) {

    _nAttributes = data->inputs();
    _nClasses = data->classes(0);

    Eigen::MatrixXd *in = data->getInputs()->getData();
    Eigen::MatrixXd *out = data->getOutputs()->getData();
    unsigned nInstances = data->instances();

    _prior.resize(_nClasses);
    _prior.setZero();

    _likelihood.resize(_nAttributes);
    for (unsigned a = 0; a < _nAttributes; ++a) {
        _likelihood[a].resize(data->categories(a), _nClasses);
        _likelihood[a].setZero();
    }

    for (unsigned i = 0; i < nInstances; ++i) {
        _prior[(unsigned)(out->coeff(i, 0))]++;
        for (unsigned a = 0; a < _nAttributes; ++a) {
            _likelihood[a].coeffRef((unsigned) (in->coeff(i, a)),
                                    (unsigned) (out->coeff(i, 0)))++;
        }
    }

    _evidence.resize(_nAttributes);
    for (unsigned a = 0; a < _nAttributes; ++a) {
        _evidence[a].resize(data->categories(a));
        _evidence[a] = _likelihood[a].rowwise().sum();
        _evidence[a] /= nInstances;
    }

    // zero-frequency problem
    bool zfp = false;
    for (unsigned a = 0; a < _nAttributes; ++a)
        for (unsigned r = 0; r < data->categories(a); ++r)
            for (unsigned c = 0; c < _nClasses; ++c)
                if (_likelihood[a].coeff(r, c) == 0)
                    zfp = true;

    if (zfp)
        for (unsigned a = 0; a < _nAttributes; ++a)
            _likelihood[a].array() += 1.0;

    for (unsigned a = 0; a < _nAttributes; ++a) {
        for (unsigned c = 0; c < _nClasses; ++c) {
            if (zfp) _likelihood[a].col(c) /= _prior[c] + data->categories(a);
            else _likelihood[a].col(c) /= _prior[c];
        }
    }

    _prior.array() /= nInstances;

}

Eigen::VectorXd metis::MultinomialNB::findEvidence(unsigned attr, Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();
    Eigen::VectorXd predictorPriorProbabilities(nInstances);

    for (unsigned i = 0; i < nInstances; ++i)
        predictorPriorProbabilities.coeffRef(i) = _evidence[attr].coeff(data->coeff(i, attr));

    return predictorPriorProbabilities;

}

Eigen::VectorXd metis::MultinomialNB::findLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const {

    unsigned nInstances = data->rows();
    Eigen::VectorXd likelihoods(nInstances);

    for (unsigned i = 0; i < nInstances; ++i)
        likelihoods.coeffRef(i) = _likelihood[attr].coeff(data->coeff(i, attr), givenClass);

    return likelihoods;

}

Eigen::VectorXd metis::MultinomialNB::findLogLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const {

    unsigned nInstances = data->rows();
    Eigen::VectorXd logLikelihoods(nInstances);

    for (unsigned i = 0; i < nInstances; ++i)
        logLikelihoods.coeffRef(i) = log(_likelihood[attr].coeff(data->coeff(i, attr), givenClass));

    return logLikelihoods;

}

metis::MultinomialNB::MultinomialNB() {

}

metis::MultinomialNB::~MultinomialNB() {

}
