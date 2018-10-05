//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/LogisticRegression.h"

namespace {

    Eigen::MatrixXd *g_in;
    Eigen::MatrixXd *g_out;
    unsigned g_nInstances;
    unsigned g_nBatches;

    metis::DataLabeled *g_test;

}

void metis::LogisticRegression::batchGradientDescent(unsigned c) {

//    Eigen::VectorXd prevLogLikelihoods(g_nInstances);
//    Eigen::VectorXd likelihoods(g_nInstances);

    Eigen::VectorXd likelihood(_batchSize);
    Eigen::VectorXd dw(_nAttributes);
    double db;

    for (unsigned i = 0; i < _iterations; ++i) {

        for (unsigned b = 0; b < g_nBatches; ++b) {

            likelihood = g_in->block(b * _batchSize, 0, _batchSize, _nAttributes) * _coeff.row(c).transpose();
            likelihood.array() += _intercept.coeff(c);

            likelihood.array() *= -1;
            likelihood = likelihood.array().exp();
            likelihood.array() += 1.0;
            likelihood = likelihood.array().inverse();

            likelihood -= g_out->block(b * _batchSize, c, _batchSize, 1);

            dw = likelihood.transpose() * g_in->block(b * _batchSize, 0, _batchSize, _nAttributes);
            dw.array() /= _batchSize;
            db = likelihood.sum();
            db /= _batchSize;

            _coeff.row(c) -= _learnRate * dw.transpose();
            _intercept.coeffRef(c) -= _learnRate * db;

        }

    }

}

void metis::LogisticRegression::stochasticGradientDescent(unsigned c) {

//    Eigen::VectorXd prevLogLikelihoods(g_nInstances);
//    Eigen::VectorXd likelihoods(g_nInstances);

    double likelihood;
    Eigen::VectorXd dw(_nAttributes);
    double db;

    for (unsigned i = 0; i < _iterations; ++i) {

        for (unsigned r = 0; r < g_nInstances; ++r) {

            likelihood = g_in->row(r) * _coeff.row(c).transpose().matrix();
            likelihood += _intercept.coeff(c);

            likelihood *= -1;
            likelihood = std::exp(likelihood);
            likelihood += 1.0;
            likelihood = 1 / likelihood;

            likelihood -= g_out->coeff(r, c);

            dw = likelihood * g_in->row(r);
            db = likelihood;

            _coeff.row(c) -= _learnRate * dw.transpose();
            _intercept.coeffRef(c) -= _learnRate * db;

        }

        if (_verbose && (i%10 == 0))
            std::cout << "Score after update #" << i << ": " << score(g_test) << std::endl;

    }

}

double metis::LogisticRegression::fit(DataLabeled *trainData, DataLabeled *testData, bool verbose) {

    _nAttributes = trainData->inputs();
    _nClasses = trainData->outputs();

    _verbose = verbose;

    g_in = trainData->getInputs()->getData();
    g_out = trainData->getOutputs()->getData();

    g_nInstances = trainData->instances();
    if (_batchSize == 0) _batchSize = g_nInstances;
    g_nBatches = g_nInstances / _batchSize;

    g_test = testData;

    DataSet outData;
    if (_nClasses == 1) {
        outData = *(trainData->getOutputs());
        outData.convertToBinaryAttributes();
        _nClasses = outData.attributes();
        g_out = outData.getData();
    }

    _coeff.resize(_nClasses, _nAttributes);
    _coeff.setRandom();
    _intercept.resize(_nClasses);
    _intercept.setRandom();

    double bestScore;

#pragma omp parallel for
    for (unsigned c = 0; c < _nClasses; ++c) {
        if (_batchSize == 1) stochasticGradientDescent(c);
        else batchGradientDescent(c);
    }

    if (_verbose)
        std::cout << "Score training: " << score(g_test) << std::endl;

    return score(testData);

}

double metis::LogisticRegression::fit(metis::DataLabeled *trainData, metis::DataLabeled *testData) {
    return fit(trainData, testData, false);
}

double metis::LogisticRegression::fit(metis::DataLabeled *trainData) {
    return fit(trainData, trainData, false);
}

Eigen::MatrixXd metis::LogisticRegression::predictProbabilities(Eigen::MatrixXd *data) const {

    Eigen::MatrixXd probabilities(data->rows(), _nClasses);

    probabilities = (*data) * _coeff.transpose();
    probabilities.rowwise() += _intercept.transpose();

    probabilities.array() *= -1;
    probabilities = probabilities.array().exp();
    probabilities.array() += 1.0;
    probabilities = probabilities.array().inverse();

    return probabilities;

}

Eigen::MatrixXd metis::LogisticRegression::predictProbabilities(metis::DataSet *data) const {
    return predictProbabilities(data->getData());
}

Eigen::VectorXi metis::LogisticRegression::predict(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();

    Eigen::MatrixXd probabilities(nInstances, _nClasses);
    probabilities = (*data) * _coeff.transpose();
    probabilities.rowwise() += _intercept.transpose();

    Eigen::VectorXi prediction(nInstances);
    double predictedLogLikelihood;

    for (unsigned i = 0; i < nInstances; ++i) {
        prediction.coeffRef(i) = 0;
        predictedLogLikelihood = probabilities.coeff(i, 0);
        for (unsigned c = 1; c < _nClasses; ++c) {
            if (probabilities.coeff(i, c) > predictedLogLikelihood) {
                prediction.coeffRef(i) = c;
                predictedLogLikelihood = probabilities.coeff(i, c);
            }
        }
    }

    return prediction;

}

Eigen::VectorXi metis::LogisticRegression::predict(metis::DataSet *data) const {
    return predict(data->getData());
}

double metis::LogisticRegression::score(metis::DataLabeled *data) const {

    Eigen::VectorXi predictions(data->instances());
    predictions = predict(data->getInputs()->getData());

    unsigned correct = 0;
    for (unsigned i = 0; i < data->instances(); ++i)
        if (data->getOutputs()->getData()->coeff(i) == predictions.coeff(i))
            ++correct;

    return (double)correct / (double)data->instances();

}

metis::LogisticRegression::LogisticRegression(unsigned iterations, double learnRate, unsigned batchSize) {

    _iterations = iterations;
    _learnRate = learnRate;
    _batchSize = batchSize;

}

Eigen::MatrixXd metis::LogisticRegression::getCoefficients() {
    return _coeff;
}

Eigen::VectorXd metis::LogisticRegression::getIntercepts() {
    return _intercept;
}

metis::LogisticRegression::LogisticRegression() {

}

metis::LogisticRegression::~LogisticRegression() {

}
