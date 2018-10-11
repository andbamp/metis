//
// Created by Andreas Bampouris on 11/10/2018.
//

#include "Classifier.h"

Eigen::VectorXi metis::Classifier::predict(Eigen::MatrixXd *input) const {
    
    unsigned nInstances = input->rows();
    Eigen::MatrixXi prediction(nInstances);
    
    // Calculates the probability of each instance belonging to each class.
    Eigen::MatrixXd probabilities = predictProbabilities(input);
    
    // Applies the one-vs.-rest classification strategy,
    // ie. chooses the class with maximum probability for each instance.
    double predictedProbability;
    for (unsigned i = 0; i < nInstances; ++i) {
        prediction.coeffRef(i) = 0;
        predictedProbability = probabilities.coeff(i, 0);
        for (unsigned c = 1; c < _nClasses; ++c) {
            if (probabilities.coeff(i, c) > predictedProbability) {
                prediction.coeffRef(i) = c;
                predictedProbability = probabilities.coeff(i, c);
            }
        }
    }
    
    return prediction;
    
}

double metis::Classifier::score(Eigen::MatrixXd *input, Eigen::VectorXi *target) const {
    
    unsigned nInstances = input->rows();
    Eigen::VectorXi predictions(nInstances);
    
    // Determines class of each instance.
    predictions = predict(input);
    
    // Compares predicted results with target values.
    unsigned correct = 0;
    for (unsigned i = 0; i < nInstances; ++i) {
        if (target->coeff(i) == predictions.coeff(i)) {
            ++correct;
        }
    }
    
    return (double)correct / (double)nInstances;
    
}
