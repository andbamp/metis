//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Regressor.h"

template <class I>
Eigen::VectorXd metis::Regressor<I>::findMSE(I *input, Eigen::MatrixXd *target) const {
    
    // Finds regression predictions.
    Eigen::MatrixXd prediction = predict(input);
    
    // Calculates mean-squared error from target values for each output variable.
    prediction -= *target;
    prediction = prediction.array().square();
    
    return prediction.colwise().mean();
    
}

template <class I>
double metis::Regressor<I>::score(I *input, Eigen::MatrixXd *target) const {
    
    // Finds mean-squared errors for each output variable and returns mean.
    return findMSE(input, target).mean();
    
}

template class metis::Regressor<Eigen::MatrixXd>;
template class metis::Regressor<Eigen::MatrixXi>;
