//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_FREQUENCY_H
#define METIS_FREQUENCY_H


#include "Probability.h"
#include <vector>

namespace metis {

class Frequency : public Probability<Eigen::MatrixXi> {

private:

    // Data
    
    //! Frequency for each attribute.
    std::vector<Eigen::VectorXd> _frequencies;
    
public:
    
    // Access
    
    /**
     * Calculates frequencies of each distinct values of the categorical attributes.
     *
     * @param data Data for which the probabilistic model is built.
     */
    void fit(Eigen::MatrixXi *data);
    
    /**
     * Calculates probabilities of input data based on frequencies.
     *
     * @param data Data the probabilities of which are calculated.
     * @return MatrixXd for probabilities of values of each attribute.
     */
    Eigen::MatrixXd findProbability(Eigen::MatrixXi *data) const;
    Eigen::VectorXd findProbability(Eigen::MatrixXi *data, unsigned attr) const;
    
    /**
     * Getter of the frequency of one particular value of one attribute.
     *
     * @param value Distinct categorical value.
     * @param attribute Attribute it refers to.
     * @return Frequency value in the requested index of _frequencies.
     */
    double getFrequency(unsigned value, unsigned attribute);
    double getFrequency(unsigned value);
    
};

}


#endif //METIS_FREQUENCY_H
