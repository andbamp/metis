//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MULTINOMIALNAIVEBAYES_H
#define METIS_MULTINOMIALNAIVEBAYES_H


#include "../src/core/Classifier.h"
#include "../src/probabilities/Frequency.h"
#include "../src/probabilities/BayesianModel.h"

namespace metis {

class MultinomialNaiveBayes : public Classifier<Eigen::MatrixXi>,
                              public BayesianModel<Eigen::MatrixXi, Frequency> {
    
private:

    // Data

public:
    
    // Access
    
    void fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target,
             Eigen::MatrixXi *valInput, Eigen::ArrayXi *valTarget, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target, unsigned verboseCycle) override;
    void fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target,
             Eigen::MatrixXi *valInput, Eigen::ArrayXi *valTarget) override;
    void fit(Eigen::MatrixXi *input, Eigen::ArrayXi *target) override;
    
    Eigen::MatrixXd predictProbabilities(Eigen::MatrixXi *input) const override;
    
    Eigen::MatrixXd findPosterior(Eigen::MatrixXi *data) const;
    Eigen::VectorXd findLikelihood(Eigen::MatrixXi *data, unsigned attr, unsigned condition) const;
    Eigen::VectorXd findEvidence(Eigen::MatrixXi *data) const;

};

}


#endif //METIS_MULTINOMIALNAIVEBAYES_H
