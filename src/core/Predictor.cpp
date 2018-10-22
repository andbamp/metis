//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Predictor.h"
#include <Eigen/Dense>

template <class I, class T>
void metis::Predictor<I, T>::fit(I *input, T *target, unsigned verboseCycle) {
    fit(input, target, nullptr, nullptr, verboseCycle);
}

template <class I, class T>
void metis::Predictor<I, T>::fit(I *input, T *target, I *valInput, T *valTarget) {
    fit(input, target, valInput, valTarget, 0);
}

template <class I, class T>
void metis::Predictor<I, T>::fit(I *input, T *target) {
    fit(input, target, nullptr, nullptr, 0);
}

template class metis::Predictor<Eigen::MatrixXd, Eigen::MatrixXd>;
template class metis::Predictor<Eigen::MatrixXi, Eigen::MatrixXd>;
template class metis::Predictor<Eigen::MatrixXd, Eigen::ArrayXi>;
template class metis::Predictor<Eigen::MatrixXi, Eigen::ArrayXi>;
