//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LINEARREGRESSION_H
#define METIS_LINEARREGRESSION_H


#include "../src/core/Regressor.h"
#include "../src/core/Optimizer.h"

namespace metis {

class LinearRegression : public Regressor<Eigen::MatrixXd>, public Optimizer {

private:

    // Meta-data
    //! Determines if an iterative method (ie. gradient descent) will be used instead of OLS.
    bool _iterative = false;
    
    // Helper methods
    /**
     * Implements the ordinary least squares method of calculating the linear model's parameters.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     */
    void ordinaryLeastSquares(Eigen::MatrixXd *input, Eigen::MatrixXd *target);
    void simpleLinearRegression(Eigen::MatrixXd *input, Eigen::MatrixXd *target);

public:

    // Access
    /**
     * Fits a linear model to a certain set of labeled data.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     * @param valInput Input data of validation set.
     * @param valTarget Output data of validation set.
     * @param verboseCycle Number of updates before user is informed for training state.
     */
    void fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
             Eigen::MatrixXd *valInput, Eigen::MatrixXd *valTarget, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
             Eigen::MatrixXd *valInput, Eigen::MatrixXd *valTarget) override;
    void fit(Eigen::MatrixXd *input, Eigen::MatrixXd *target) override;
    
    /**
     * General method for predicting output for a given input.
     *
     * @param input Input data. Each row represents an instance.
     */
    Eigen::MatrixXd predict(Eigen::MatrixXd *input) const override;
    
    // Construction
    /**
     * Creates a linear regressor.
     *
     * @param iterative If true, gradient descent is used to determine the linear model's parameters.
     *                  If false, the analytic method of ordinary least squares is used.
     * @param iterations Maximum number of iterations. Applicable only in the case of iterative method being used.
     * @param learnRate Learn rate for gradient descent. Applicable only in the case of iterative method being used.
     * @param batchSize Size of the mini-batch. If set to 0, the full batch is used.
     *                  Applicable only in the case of iterative method being used.
     */
    LinearRegression(bool iterative, unsigned iterations, double learnRate, unsigned batchSize);
    LinearRegression();
    ~LinearRegression();
    
};

}


#endif //METIS_LINEARREGRESSION_H
