//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_NAIVEBAYES_H
#define METIS_NAIVEBAYES_H

#include "DataLabeled.h"

namespace metis {

template <class T>
class NaiveBayes {

protected:

    // 3. Structure
    Eigen::VectorXd _prior; // class prior P(c)

    // 4. State
    unsigned _nAttributes;
    unsigned _nClasses;

    // 5. Helper methods

public:

    // 2. Interface methods
    virtual void fit(DataLabeled *data) = 0;

    Eigen::VectorXd findPrior(Eigen::MatrixXd *data) const;
    virtual Eigen::VectorXd findEvidence(unsigned attr, Eigen::MatrixXd *data) const = 0;
    virtual Eigen::VectorXd findLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const = 0;
    virtual Eigen::VectorXd findLogLikelihood(unsigned attr, Eigen::MatrixXd *data, unsigned givenClass) const = 0;

    Eigen::MatrixXd findPosterior(Eigen::MatrixXd *data) const;
    Eigen::MatrixXd findRelativePosterior(Eigen::MatrixXd *data) const;
    Eigen::MatrixXd findLogPosterior(Eigen::MatrixXd *data) const;

    Eigen::VectorXi predict(Eigen::MatrixXd *data, bool logPost) const;
    Eigen::VectorXi predict(Eigen::MatrixXd *data) const;

    double score(DataLabeled *data, bool logPost) const;
    double score(DataLabeled *data) const;

    // 1. Construction

};

}


#endif //METIS_NAIVEBAYES_H
