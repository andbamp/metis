//
// Created by Andreas Bampouris on 11/10/2018.
//

#ifndef METIS_LINEARREGRESSIONNEW_H
#define METIS_LINEARREGRESSIONNEW_H


#include "../src/core/Regressor.h"
#include "../src/core/Optimizer.h"

namespace metis {

class LinearRegressionNew : public Regressor, public Optimizer {

private:

    // Meta-data
    //! Determines if an iterative method (ie. gradient descent) will be used instead of OLS.
    bool _iterative = false;
    
    // Helper methods
    void ordinaryLeastSquares(Eigen::MatrixXd *input, Eigen::MatrixXd *target);
    void simpleLinearRegression(Eigen::MatrixXd *input, Eigen::MatrixXd *target);

public:

    // Access
    /**
     * Fitting a linear model to a certain set of labeled data.
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
    LinearRegressionNew(bool iterative, unsigned iterations, double learnRate, unsigned batchSize);
    LinearRegressionNew();
    ~LinearRegressionNew();
    
};

}


#endif //METIS_LINEARREGRESSIONNEW_H
