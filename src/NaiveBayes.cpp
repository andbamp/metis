//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/NaiveBayes.h"

template<class T>
Eigen::VectorXd metis::NaiveBayes<T>::findPrior(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();
    Eigen::VectorXd classPriorProbabilities(nInstances);

    for (unsigned i = 0; i < nInstances; ++i)
        classPriorProbabilities.coeffRef(i) = _prior.coeff(data->coeff(i, 0));

    return classPriorProbabilities;

}

template<class T>
Eigen::MatrixXd metis::NaiveBayes<T>::findPosterior(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();
    Eigen::MatrixXd probabilities(_nClasses, nInstances);
    probabilities = findRelativePosterior(data);

    for (unsigned i = 0; i < nInstances; ++i)
        probabilities.col(i) /= probabilities.col(i).sum();

    return probabilities;

}

template<class T>
Eigen::MatrixXd metis::NaiveBayes<T>::findRelativePosterior(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();
    Eigen::MatrixXd probabilities(_nClasses, nInstances);
    probabilities.setOnes();

    for (unsigned c = 0; c < _nClasses; ++c) {
        for (unsigned a = 0; a < _nAttributes; ++a)
            probabilities.array().row(c) *= findLikelihood(a, data, c).transpose().array();
        probabilities.array().row(c) *= _prior.coeff(c);
    }

    return probabilities;

}

template<class T>
Eigen::MatrixXd metis::NaiveBayes<T>::findLogPosterior(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();
    Eigen::MatrixXd logProbabilities(_nClasses, nInstances);
    logProbabilities.setZero();

    for (unsigned c = 0; c < _nClasses; ++c) {
        for (unsigned a = 0; a < _nAttributes; ++a)
            logProbabilities.array().row(c) += findLogLikelihood(a, data, c).transpose().array();
        logProbabilities.array().col(c) += log(_prior.coeff(c));
    }

    return logProbabilities;

}

template<class T>
Eigen::VectorXi metis::NaiveBayes<T>::predict(Eigen::MatrixXd *data, bool logPost) const {

    unsigned nInstances = data->rows();

    Eigen::MatrixXd probabilities(_nClasses, nInstances);

    if (logPost) probabilities = findLogPosterior(data);
    else probabilities = findRelativePosterior(data);

    Eigen::VectorXi predictions(nInstances);
    double predictedProbability;

    for (unsigned i = 0; i < nInstances; ++i) {
        predictions.coeffRef(i) = 0;
        predictedProbability = probabilities.coeff(0, i);
        for (unsigned c = 1; c < _nClasses; ++c) {
            if (probabilities.coeff(c, i) > predictedProbability) {
                predictions.coeffRef(i) = c;
                predictedProbability = probabilities.coeff(c, i);
            }
        }
    }

    return predictions;

}

template<class T>
Eigen::VectorXi metis::NaiveBayes<T>::predict(Eigen::MatrixXd *data) const {
    return predict(data, false);
}

template<class T>
double metis::NaiveBayes<T>::score(DataLabeled *data, bool logPost) const {

    Eigen::VectorXi predictions(data->instances());
    predictions = predict(data->getInputs()->getData(), logPost);

    unsigned correct = 0;
    for (unsigned i = 0; i < data->instances(); ++i)
        if (data->getOutputs()->getData()->coeff(i) == predictions.coeff(i))
            ++correct;

    return (double)correct / (double)data->instances();

}

template<class T>
double metis::NaiveBayes<T>::score(metis::DataLabeled *data) const {
    return score(data, false);
}

template class metis::NaiveBayes<std::string>;
template class metis::NaiveBayes<double>;
