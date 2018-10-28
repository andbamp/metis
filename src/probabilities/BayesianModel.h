//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_BAYESIAN_H
#define METIS_BAYESIAN_H

#include "Frequency.h"

namespace metis {

template <class D, class P>
class BayesianModel {

protected:

    // Data
    
    //! The prior probability of a certain condition expressed as a frequency.
    //! (ie. a class in the case of the Naive Bayes classifier).
    Frequency _prior;
    
    //! The conditional probability of each predictor expressed with as many frequencies or distributions as there are
    //! possible conditions. (Condition being a specific class in the Naive Bayes classifier).
    std::vector<P *> _likelihood;
    
    //! The prior probability of each predictor.
    P _evidence;
    
    // Meta-data
    
    //! Number of conditions, determining the length of _likelihood vector.
    unsigned _nConditions = 1;

public:
    
    // Access
    
    /**
     * Determines needed probabilities.
     *
     * @param data Training data.
     */
    void determineProbabilities(D *input, Eigen::ArrayXi *target);
    
    /**
     * Calculates posterior probabilities of each instance belonging on each possible condition (ie. each class).
     *
     * @param data Data of type either MatrixXd or MatrixXi. Each row represents a different instance.
     * @return MatrixXd with as many columns as there are classes.
     */
    Eigen::MatrixXd findPosterior(D *data) const;
    
    /**
     * Gets prior probability of each condition.
     *
     * @param data 1-D array of integers representing conditions (ie. classes) of instances.
     * @return VectorXd with the probabilities of each condition.
     */
    Eigen::VectorXd findPrior(Eigen::ArrayXi *data) const;
    
    /**
     * Gets prior probability of a specific condition.
     *
     * @param condition Specific condition.
     * @return Value of probability of given condition.
     */
    Eigen::VectorXd findPrior(unsigned condition) const;
    
    /**
     * Gets conditional probability for one specific attribute of each instance given one specific condition.
     *
     * @param data Data of type either MatrixXd or MatrixXi. Each row represents a different instance.
     * @param attr Index of target attribute.
     * @param condition Index of given condition.
     * @return VectorXd with as coefficients as there are instances.
     */
    Eigen::VectorXd findLikelihood(D *data, unsigned attr, unsigned condition) const;
    
    /**
     * Gets prior probability the value of each instance.
     *
     * @param data Data of type either MatrixXd or MatrixXi. Each row represents a different instance.
     * @return VectorXd with as coefficients as there are instances.
     */
    Eigen::VectorXd findEvidence(D *data) const;

};

}


#endif //METIS_BAYESIAN_H
