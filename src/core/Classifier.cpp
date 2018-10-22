//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Classifier.h"

template <class I>
Eigen::ArrayXi metis::Classifier<I>::predict(I *input) const {
    
    // Calculates the probability of each instance belonging to each class.
    Eigen::MatrixXd probabilities = predictProbabilities(input);
    
    unsigned nInstances = probabilities.rows();
    Eigen::ArrayXi prediction(nInstances);
    
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

template <class I>
double metis::Classifier<I>::score(I *input, Eigen::ArrayXi *target) const {
    
    // Determines class of each instance.
    Eigen::ArrayXi predictions = predict(input);
    
    unsigned nInstances = predictions.rows();
    
    // Compares predicted results with target values.
    unsigned correct = 0;
    for (unsigned i = 0; i < nInstances; ++i) {
        if (target->coeff(i) == predictions.coeff(i)) {
            ++correct;
        }
    }
    
    return (double)correct / (double)nInstances;
    
}

template class metis::Classifier<Eigen::MatrixXd>;
template class metis::Classifier<Eigen::MatrixXi>;
