//
// Created by Andreas Bampouris on 13/10/2018.
//

#ifndef METIS_LOGISTICREGRESSIONNEW_H
#define METIS_LOGISTICREGRESSIONNEW_H


#include "../src/core/Classifier.h"
#include "../src/core/Optimizer.h"

namespace metis {

class LogisticRegressionNew : public Classifier, public Optimizer {

private:


public:

    // Access
    /**
     * Fits a logistic model to a certain set of labeled data.
     *
     * @param input Input data of training set. Each row represents an instance.
     * @param target Output data of training set. Same number of rows as input.
     * @param valInput Input data of validation set.
     * @param valTarget Output data of validation set.
     * @param verboseCycle Number of updates before user is informed for training state.
     */
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
             Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
             Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target) override;
    
    Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *input) const override;
    
    // Construction
    /**
     * Creates a logistic regression classifier.
     *
     * @param iterations Maximum number of iterations.
     * @param learnRate Learn rate for gradient descent.
     * @param batchSize Size of the mini-batch. If set to 0, the full batch is used.
     */
    LogisticRegressionNew(unsigned iterations, double learnRate, unsigned batchSize);
    LogisticRegressionNew();
    ~LogisticRegressionNew();
    
};

}


#endif //METIS_LOGISTICREGRESSIONNEW_H
