//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_PREDICTOR_H
#define METIS_PREDICTOR_H


#include <Eigen/Dense>

namespace metis {

template <class T>
class Predictor {

protected:

    // Meta-data
    
    //! Number of input attributes, ie. number of columns on each input vector.
    unsigned _nAttributes;

public:

    // Access
    
    /**
     * General method for fitting a model to a certain set of labeled data.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     * @param valInput Input data of validation set.
     * @param valTarget Output data of validation set.
     * @param verboseCycle Number of updates before user is informed for training state.
     */
    virtual void fit(Eigen::MatrixXd *input, T *target, Eigen::MatrixXd *valInput, T *valTarget,
                     unsigned verboseCycle) = 0;
    virtual void fit(Eigen::MatrixXd *input, T *target, unsigned verboseCycle);
    virtual void fit(Eigen::MatrixXd *input, T *target, Eigen::MatrixXd *valInput, T *valTarget);
    virtual void fit(Eigen::MatrixXd *input, T *target);
    
    /**
     * General method for predicting output for a given input.
     *
     * @param input Input data. Each row represents an instance.
     */
    virtual T predict(Eigen::MatrixXd *input) const = 0;
    
    /**
     * General method for assessing correctness.
     *
     * @param input Input data of test set. Each row represents an instance.
     * @param target Output data of test set. Same number of rows as input.
     */
    virtual double score(Eigen::MatrixXd *input, T *target) const = 0;

};

}


#endif //METIS_PREDICTOR_H
