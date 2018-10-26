//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_PROBABILITY_H
#define METIS_PROBABILITY_H


#include <Eigen/Dense>

namespace metis {

template <class I>
class Probability {

protected:

    // Data
    
    // Meta-data
    
    
public:

    // Access
    
    /**
     * General method for determining the probabilistic model.
     *
     * @param data Data for which the probabilistic model is built.
     */
    virtual void fit(I *data) = 0;
    
    /**
     * General method for calculating probabilities of input data.
     *
     * @param data Data the probabilities of which are calculated.
     * @return MatrixXd for probabilities of values of each attribute.
     */
    virtual Eigen::MatrixXd findProbability(I *data) const = 0;
    virtual Eigen::VectorXd findProbability(I *data, unsigned attr) const = 0;

};

}


#endif //METIS_PROBABILITY_H
