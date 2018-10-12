//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/LinearRegression.h"


void metis::LinearRegression::ordinaryLeastSquares(Eigen::MatrixXd *input, Eigen::MatrixXd *target) {

    for (unsigned o = 0; o < _nOutputs; ++o) {
        _coeff.row(o) = ( (input->transpose() * (*input)).inverse() * (input->transpose()) * target->col(o)
                ).transpose();
        _intercept.coeffRef(o) = target->col(o).mean();
    }

}

void metis::LinearRegression::simpleLinearRegression(Eigen::MatrixXd *input, Eigen::MatrixXd *target) {
    
    double meanIn = input->col(0).mean();
    
    for (unsigned o = 0; o < _nOutputs; ++o) {
        
        double meanOut = target->col(o).mean();
        
        double numerator = ((input->array() - meanIn).array() * target->col(o).array()).sum();
        double denominator = (input->array() - meanIn).array().square().sum();
        
        _coeff.coeffRef(o, 0) = numerator / denominator;
        _intercept.coeffRef(o) = meanOut - _coeff.coeffRef(o, 0) * meanIn;
        
    }

}

void metis::LinearRegression::fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
                                     Eigen::MatrixXd *valInput, Eigen::MatrixXd *valTarget, unsigned verboseCycle) {
    
    _nAttributes = input->cols();
    _nOutputs = target->cols();
    
    _coeff.resize(_nOutputs, _nAttributes);
    _intercept.resize(_nOutputs);
    
    if (_iterative) {
    
        _nModels = _nOutputs;
        _coeff.setRandom();
        _intercept.setRandom();
        
        if (_batchSize == 1) stochasticGradientDescent(input, target, 0);
        else batchGradientDescent(input, target, 0);
        
    } else {
    
        // Simple linear regression is a specialization of OLS that runs when there's only one attribute.
        if (_nAttributes == 1) simpleLinearRegression(input, target);
        else ordinaryLeastSquares(input, target);
    
    }
    
}

void metis::LinearRegression::fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target, unsigned verboseCycle) {
    Predictor::fit(input, target, verboseCycle);
}

void metis::LinearRegression::fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
                                     Eigen::MatrixXd *valInput, Eigen::MatrixXd *valTarget) {
    Predictor::fit(input, target, valInput, valTarget);
}

void metis::LinearRegression::fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target) {
    Predictor::fit(input, target);
}

Eigen::MatrixXd metis::LinearRegression::predict(Eigen::MatrixXd *input) const {
    
    Eigen::MatrixXd predictions(input->rows(), _nModels);
    
    // Prediction is calculated as a linear transformation of the inputs.
    predictions = (*input) * _coeff.transpose();
    predictions.rowwise() += _intercept.transpose();
    
    return predictions;
    
}

metis::LinearRegression::LinearRegression(bool iterative, unsigned iterations,
                                                double learnRate, unsigned batchSize) {
    
    _iterative = iterative;
    _iterations = iterations;
    _learnRate = learnRate;
    _batchSize = batchSize;
    
}

metis::LinearRegression::LinearRegression() {

}

metis::LinearRegression::~LinearRegression() {

}
