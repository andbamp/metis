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
    
    // Meta-data
    
    unsigned _nInstances;
    Eigen::ArrayXi _nCategories;

public:
    
    // Access
    
    /**
     * Calculates frequencies of each distinct value of the categorical attributes.
     *
     * @param data Data for which the probabilistic model is built.
     */
    template <class I>
    void fit(I *data);
    void fit(Eigen::MatrixXi *data) override;
    void fit(Eigen::VectorXi *data);
    void fit(Eigen::ArrayXi *data);
    
    /**
     * Resolves the so-called zero-frequency problem. Useful in algorithms such as Naive Bayes.
     *
     * @return
     */
    void eliminateZeroFrequencies();
    
    /**
     * Inspects frequencies to see if any of them equals to zero.
     *
     * @return True or false based on assessment.
     */
    bool zeroFrequencyExists();
    
    /**
     * Calculates probabilities of input data based on frequencies.
     *
     * @param data Data the probabilities of which are calculated.
     * @return MatrixXd for probabilities of values of each attribute.
     */
    Eigen::MatrixXd findProbability(Eigen::MatrixXi *data) const override;
    Eigen::VectorXd findProbability(Eigen::MatrixXi *data, unsigned attr) const override;
    
    /**
     * Getter of the frequency of one particular value of one attribute.
     *
     * @param value Distinct categorical value.
     * @param attribute Attribute it refers to.
     * @return Frequency value in the requested index of _frequencies.
     */
    double getFrequency(unsigned value, unsigned attribute) const;
    double getFrequency(unsigned value) const;
    
};

}


#endif //METIS_FREQUENCY_H
