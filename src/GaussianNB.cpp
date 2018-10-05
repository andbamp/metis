//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/GaussianNB.h"

void metis::GaussianNB::fit(metis::DataLabeled *data) {

    _nAttributes = data->inputs();
    _nClasses = data->classes(0);

    Eigen::MatrixXd *in = data->getInputs()->getData();
    Eigen::MatrixXd *out = data->getOutputs()->getData();
    unsigned nInstances = data->instances();

    std::vector<std::vector<unsigned>> examplesInClass(_nClasses);
    for (unsigned i = 0; i < nInstances; ++i)
        examplesInClass[(unsigned)(out->coeff(i, 0))].push_back(i);

    _prior.resize(_nClasses);
    _means.resize(_nAttributes, _nClasses);
    _stDev.resize(_nAttributes, _nClasses);

    for (unsigned c = 0; c < _nClasses; ++c) {

        _prior.coeffRef(c) = examplesInClass[c].size();
        _prior.coeffRef(c) /= nInstances;

        Eigen::MatrixXd dataInClass(examplesInClass[c].size(), _nAttributes);
        for (unsigned e = 0; e < examplesInClass[c].size(); ++e)
            dataInClass.row(e) = in->row(examplesInClass[c][e]);

        for (unsigned a = 0; a < _nAttributes; ++a) {
            _means.coeffRef(a, c) = dataInClass.col(a).mean();
            _stDev.coeffRef(a, c) =
                    std::sqrt((dataInClass.array().col(a) - _means.coeffRef(a, c)).square().sum() /
                              (examplesInClass[c].size() - 1));
        }

    }

}

Eigen::VectorXd metis::GaussianNB::findEvidence(unsigned attr, Eigen::MatrixXd *data) const {

    Eigen::VectorXd evidences(data->rows());

//    evidences.array() = (data->col(attr).array() - _means.coeff(attr, givenClass)).square();
//    evidences.array() /= -2 * pow(_stDev.coeff(attr, givenClass), 2);
//    evidences.array() = evidences.array().exp();
////    evidences.array() /= sqrt(pow(2 * M_PI, _nAttributes) * pow(_stDev.coeff(attr, givenClass), 2));
//    evidences.array() /= sqrt(2 * M_PI * pow(_stDev.coeff(attr, givenClass), 2));

    return evidences;

}

Eigen::VectorXd metis::GaussianNB::findLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const {

    Eigen::VectorXd likelihoods(data->rows());

    likelihoods.array() = (data->col(attr).array() - _means.coeff(attr, givenClass)).square();
    likelihoods.array() /= -2 * pow(_stDev.coeff(attr, givenClass), 2);
    likelihoods.array() = likelihoods.array().exp();
//    likelihoods.array() /= sqrt(pow(2 * M_PI, _nAttributes) * pow(_stDev.coeff(attr, givenClass), 2));
    likelihoods.array() /= sqrt(2 * M_PI * pow(_stDev.coeff(attr, givenClass), 2));

    return likelihoods;

}

Eigen::VectorXd metis::GaussianNB::findLogLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const {

    Eigen::VectorXd logLikelihoods(data->rows());

    logLikelihoods.setZero();
//    logLikelihoods.array() -= _nAttributes * log(2.0 * M_PI) / 2.0;
    logLikelihoods.array() -= log(2.0 * M_PI) / 2.0;
    logLikelihoods.array() -= log(abs(_stDev.coeff(attr, givenClass))) / 2.0;
    logLikelihoods.array() -= (data->array().col(attr) - _means.coeff(attr, givenClass)).square().array() / (2.0 * _stDev.coeff(attr, givenClass));

    return logLikelihoods;

}

metis::GaussianNB::GaussianNB() {

}

metis::GaussianNB::~GaussianNB() {

}
