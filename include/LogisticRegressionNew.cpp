//
// Created by Andreas Bampouris on 13/10/2018.
//

#include "LogisticRegressionNew.h"
#include "DataContainer.h"

void metis::LogisticRegressionNew::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
                                       Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) {
    
    // The One-vs.-Rest strategy will be used, which means the number of models trained will be equal to the number of
    // classes in the target variable. The latter needs to first be converted to as many binary outputs as classes.
    Eigen::MatrixXd binTarget = DataContainer::convertToBinaryMatrix(target);
    _nClasses = binTarget.cols();
    _nModels = _nClasses;
    _nAttributes = input->cols();
    
    _coeff.resize(_nModels, _nAttributes);
    _intercept.resize(_nModels);
    
    _coeff.setRandom();
    _intercept.setRandom();
    
    if (_batchSize == 1) stochasticGradientDescent(input, &binTarget, 1);
    else batchGradientDescent(input, &binTarget, 1);
    
}

void metis::LogisticRegressionNew::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target, unsigned verboseCycle) {
    Predictor::fit(input, target, verboseCycle);
}

void metis::LogisticRegressionNew::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
                                       Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget) {
    Predictor::fit(input, target, valInput, valTarget);
}

void metis::LogisticRegressionNew::fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target) {
    Predictor::fit(input, target);
}

Eigen::MatrixXd metis::LogisticRegressionNew::predictProbabilities(Eigen::MatrixXd *input) const {
    
    Eigen::MatrixXd predictions(input->rows(), _nModels);
    
    // Linear transformation of the input is first calculated.
    predictions = (*input) * _coeff.transpose();
    predictions.rowwise() += _intercept.transpose();
    
    // Sigmoid function of the linear transformation gives the prediction.
    predictions.array() *= -1;
    predictions.array() = predictions.array().exp();
    predictions.array() += 1.0;
    predictions.array() = predictions.array().inverse();
    
    return predictions;
    
}

metis::LogisticRegressionNew::LogisticRegressionNew(unsigned iterations, double learnRate, unsigned batchSize) {
    
    _iterations = iterations;
    _learnRate = learnRate;
    _batchSize = batchSize;
    
}

metis::LogisticRegressionNew::LogisticRegressionNew() {

}

metis::LogisticRegressionNew::~LogisticRegressionNew() {

}
