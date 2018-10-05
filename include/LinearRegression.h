//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LINEARREGRESSION_H
#define METIS_LINEARREGRESSION_H

#include "DataLabeled.h"

namespace metis {

class LinearRegression {

private:

    // 3. Structure
    Eigen::MatrixXd _coeff;
    Eigen::VectorXd _intercept;

    // 4. State
    unsigned _nAttributes;
    unsigned _nOutputs;

    // 5. Helper methods
    void simpleLinearRegression(DataLabeled *data);
    void ordinaryLeastSquares(DataLabeled *data);

public:

    // 2. Interface methods
    void fit(DataLabeled *data);

    Eigen::MatrixXd predict(Eigen::MatrixXd *data) const;
    Eigen::MatrixXd predict(DataSet *data) const;

    Eigen::VectorXd score(DataLabeled *data) const;

    Eigen::MatrixXd getCoefficients() const;
    Eigen::VectorXd getIntercept() const;

    // 1. Construction
    LinearRegression();
    ~LinearRegression();

};

}

#endif //METIS_LINEARREGRESSION_H
