//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_CLASSIFIER_H
#define METIS_CLASSIFIER_H


#include "Predictor.h"
#include <Eigen/Dense>

namespace metis {

template <class I>
class Classifier : public Predictor<I, Eigen::ArrayXi> {

protected:
    
    // Meta-data
    //! Number of possible classes for output variable.
    unsigned _nClasses;

public:
    
    // Access
    /**
     * Returns probability for each given instance to belong on each class.
     *
     * @param input Input data with unknown output.
     */
    virtual Eigen::MatrixXd predictProbabilities(I *input) const = 0;
    
    Eigen::ArrayXi predict(I *input) const override;
    
    double score(I *input, Eigen::ArrayXi *target) const override;

};

}


#endif //METIS_CLASSIFIER_H
