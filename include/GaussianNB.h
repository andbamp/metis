//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_GAUSSIANNB_H
#define METIS_GAUSSIANNB_H

#include "NaiveBayes.h"

namespace metis {

class GaussianNB : public NaiveBayes<double> {

private:

    // 3. Structure
    Eigen::MatrixXd _means;
    Eigen::MatrixXd _stDev;

    // 4. State

    // 5. Helper methods

public:

    // 2. Interface methods
    void fit(DataLabeled *data) override;

    Eigen::VectorXd findEvidence(unsigned attr, Eigen::MatrixXd *data) const override;
    Eigen::VectorXd findLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const override;
    Eigen::VectorXd findLogLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const override;

    // 1. Construction
    GaussianNB();
    ~GaussianNB();

};

}


#endif //METIS_GAUSSIANNB_H
