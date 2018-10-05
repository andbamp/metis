//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/LinearRegression.h"

void metis::LinearRegression::simpleLinearRegression(metis::DataLabeled *data) {

    Eigen::MatrixXd *in = data->getInputs()->getData();
    Eigen::MatrixXd *out = data->getOutputs()->getData();

    double meanIn = in->col(0).mean();

#pragma omp parallel for
    for (unsigned o = 0; o < _nOutputs; ++o) {

        double meanOut = out->col(o).mean();

        double numerator = ((in->array() - meanIn).array() * out->col(o).array()).sum();
        double denominator = (in->array() - meanIn).array().square().sum();

        _coeff.coeffRef(o, 0) = numerator / denominator;
        _intercept.coeffRef(o) = meanOut - _coeff.coeffRef(o, 0) * meanIn;

    }

}

void metis::LinearRegression::ordinaryLeastSquares(metis::DataLabeled *data) {

    Eigen::MatrixXd *in = data->getInputs()->getData();
    Eigen::MatrixXd *out = data->getOutputs()->getData();

#pragma omp parallel for
    for (unsigned o = 0; o < _nOutputs; ++o) {
        _coeff.col(o) = (in->transpose() * (*in)).inverse() * (in->transpose()) * out->col(o);
        _intercept.coeffRef(o) = out->col(o).mean();
    }

}

void metis::LinearRegression::fit(metis::DataLabeled *data) {

    _nAttributes = data->inputs();
    _nOutputs = data->outputs();

    _coeff.resize(_nAttributes, _nOutputs);
    _intercept.resize(_nOutputs);

    if (_nAttributes == 1) simpleLinearRegression(data);
    else ordinaryLeastSquares(data);

}

Eigen::MatrixXd metis::LinearRegression::predict(Eigen::MatrixXd *data) const {

    Eigen::MatrixXd predictions(data->rows(), _nOutputs);

    predictions = (*data) * _coeff.matrix();
    predictions.rowwise() += _intercept.transpose();

    return predictions;

}

Eigen::MatrixXd metis::LinearRegression::predict(metis::DataSet *data) const {
    return predict(data->getData());
}

Eigen::VectorXd metis::LinearRegression::score(metis::DataLabeled *data) const {

    Eigen::MatrixXd prediction(data->instances(), _nOutputs);
    prediction = predict(data->getInputs());
    prediction.array() -= data->getOutputs()->getData()->array();
    prediction.array() = prediction.array().square();

    Eigen::VectorXd predictionMeans(_nOutputs);
    predictionMeans = prediction.colwise().mean();

    return predictionMeans;

}

Eigen::MatrixXd metis::LinearRegression::getCoefficients() const {
    return _coeff;
}

Eigen::VectorXd metis::LinearRegression::getIntercept() const {
    return _intercept;
}

metis::LinearRegression::LinearRegression() {

}

metis::LinearRegression::~LinearRegression() {

}
