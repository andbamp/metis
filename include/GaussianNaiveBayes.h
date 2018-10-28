//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_NAIVEBAYESCLASSIFIER_H
#define METIS_NAIVEBAYESCLASSIFIER_H


#include "../src/core/Classifier.h"
#include "../src/probabilities/BayesianModel.h"
#include "../src/probabilities/GaussianDistribution.h"

namespace metis {

class GaussianNaiveBayes : public Classifier<Eigen::MatrixXd>,
                           public BayesianModel<Eigen::MatrixXd, GaussianDistribution> {

private:

    // Data
    

public:

    // Access
    
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
             Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target,
             Eigen::MatrixXd *valInput, Eigen::ArrayXi *valTarget) override;
    void fit(Eigen::MatrixXd *input, Eigen::ArrayXi *target) override;
    
    Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *input) const override;
    
    Eigen::MatrixXd findPosterior(Eigen::MatrixXd *data) const;
    Eigen::VectorXd findLikelihood(Eigen::MatrixXd *data, unsigned attr, unsigned condition) const;
    Eigen::VectorXd findEvidence(Eigen::MatrixXd *data) const;

};

}


#endif //METIS_NAIVEBAYESCLASSIFIER_H
