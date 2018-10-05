//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MULTINOMIALNB_H
#define METIS_MULTINOMIALNB_H

#include "NaiveBayes.h"

namespace metis {

class MultinomialNB : public NaiveBayes<std::string> {

private:

    // 3. Structure
    std::vector<Eigen::MatrixXd> _likelihood; // predictor given class P(x|c)
    std::vector<Eigen::VectorXd> _evidence; // predictor prior P(x)

    // 4. State

    // 5. Helper methods

public:

    // 2. Interface methods
    void fit(DataLabeled *data) override;

    Eigen::VectorXd findEvidence(unsigned attr, Eigen::MatrixXd *data) const override;
    Eigen::VectorXd findLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const override;
    Eigen::VectorXd findLogLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const override;

    // 1. Construction
    MultinomialNB();
    ~MultinomialNB();

};

}


#endif //METIS_MULTINOMIALNB_H
