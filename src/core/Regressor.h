//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_REGRESSOR_H
#define METIS_REGRESSOR_H


#include "Predictor.h"
#include <Eigen/Dense>

namespace metis {

template <class I>
class Regressor : public Predictor<I, Eigen::MatrixXd> {

protected:
    
    // Meta-data
    //! Number of output attributes, ie. number of columns on each output vector.
    unsigned _nOutputs;

public:

    // Access
    virtual Eigen::MatrixXd predict(I *input) const override = 0;
    
    /**
     * Returns mean-squared error for each output variable.
     *
     * @param input Input data of test set.
     * @param target Output data of test set.
     */
    Eigen::VectorXd findMSE(I *input, Eigen::MatrixXd *target) const;
    
    double score(I *input, Eigen::MatrixXd *target) const override;

};

}


#endif //METIS_REGRESSOR_H
