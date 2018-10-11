//
// Created by Andreas Bampouris on 11/10/2018.
//

#include "Regressor.h"

Eigen::VectorXd metis::Regressor::findMSE(Eigen::MatrixXd *input, Eigen::MatrixXd *target) const {
    
    // Finds regression predictions.
    Eigen::MatrixXd prediction = predict(input);
    
    // Calculates mean-squared error from target values for each output variable.
    prediction -= *target;
    prediction = prediction.array().square();
    
    return prediction.colwise().mean();
    
}

double metis::Regressor::score(Eigen::MatrixXd *input, Eigen::MatrixXd *target) const {
    
    // Finds mean-squared errors for each class and returns mean.
    return findMSE(input, target).mean();
    
}
