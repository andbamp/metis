//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_CLASSIFIER_H
#define METIS_CLASSIFIER_H


#include "Predictor.h"

namespace metis {

class Classifier : public Predictor<Eigen::ArrayXi> {

protected:
    
    // Meta-data
    //! Number of possible classes for output variable.
    unsigned _nClasses;

public:
    
    // Access
    /**
     * Returns probability for each given instance to belong on each class.
     *
     * @param input Input data. Each row represents an instance.
     */
    virtual Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *input) const = 0;
    
    Eigen::ArrayXi predict(Eigen::MatrixXd *input) const override;
    
    double score(Eigen::MatrixXd *input, Eigen::ArrayXi *target) const override;

};

}


#endif //METIS_CLASSIFIER_H
