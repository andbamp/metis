//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Predictor.h"

template<class T>
void metis::Predictor<T>::fit(Eigen::MatrixXd *input, T *target, unsigned verboseCycle) {
    fit(input, target, nullptr, nullptr, verboseCycle);
}

template<class T>
void metis::Predictor<T>::fit(Eigen::MatrixXd *input, T *target, Eigen::MatrixXd *valInput, T *valTarget) {
    fit(input, target, valInput, valTarget, 0);
}

template<class T>
void metis::Predictor<T>::fit(Eigen::MatrixXd *input, T *target) {
    fit(input, target, nullptr, nullptr, 0);
}

template class metis::Predictor<Eigen::MatrixXd>;
template class metis::Predictor<Eigen::ArrayXi>;
