//
// Created by Andreas Bampouris on 11/10/2018.
//

#ifndef METIS_REGRESSOR_H
#define METIS_REGRESSOR_H


#include "Predictor.h"

namespace metis {

class Regressor : public Predictor<Eigen::MatrixXd> {

protected:
    
    // Meta-data
    
    //! Number of output attributes, ie. number of columns on each output vector.
    unsigned _nOutputs;

public:

    // Access
    
    virtual Eigen::MatrixXd predict(Eigen::MatrixXd *input) const override = 0;
    
    /**
     * Returns mean-squared error for each output variable.
     *
     * @param input Input data of test set. Each row represents an instance.
     * @param target Output data of test set. Same number of rows as input.
     */
    Eigen::VectorXd findMSE(Eigen::MatrixXd *input, Eigen::MatrixXd *target) const;
    
    double score(Eigen::MatrixXd *input, Eigen::MatrixXd *target) const override;

};

}


#endif //METIS_REGRESSOR_H
