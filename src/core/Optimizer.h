//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_OPTIMIZER_H
#define METIS_OPTIMIZER_H


#include <Eigen/Dense>

namespace metis {

class Optimizer {

private:
    
    // Helper methods
    
    /**
     * Calculates the prediction and places it on @param prediction.
     *
     * @param prediction Linear transform of inputs.
     * @param transferFunc Transfer function whose output is being optimized.
     */
    void transferFunction(double *prediction, unsigned transferFunc);
    void transferFunction(Eigen::VectorXd *prediction, unsigned transferFunc);

protected:
    
    // Data
    //! Parameters of linear model
    Eigen::MatrixXd _coeff;
    
    //! Bias of linear model
    Eigen::VectorXd _intercept;
    
    // Meta-data
    //! Number of models to be optimized.
    unsigned _nModels;
    
    //! Number of maximum iterations.
    unsigned _iterations = 1;
    
    //! Learn rate of gradient descent.
    double _learnRate = 0.01;
    
    //! Size of mini-batch.
    unsigned _batchSize = 1;
    
    // Helper methods
    /**
     * Optimizes parameters using mini-batch gradient descent.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     * @param transferFunc Transfer function whose output is being optimized.
     */
    void batchGradientDescent(Eigen::MatrixXd *input, Eigen::MatrixXd *target, unsigned transferFunc);
    
    /**
     * Optimizes parameters using stochastic gradient descent.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     * @param transferFunc Transfer function whose output is being optimized.
     */
    void stochasticGradientDescent(Eigen::MatrixXd *input, Eigen::MatrixXd *target, unsigned transferFunc);

public:

    // Access
    /**
     *
     * @return Coefficients of linear model
     */
    Eigen::MatrixXd getCoefficients();
    
    /**
     *
     * @return Bias of linear model
     */
    Eigen::VectorXd getIntercepts();

};

}


#endif //METIS_OPTIMIZER_H
