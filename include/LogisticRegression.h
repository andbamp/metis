//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_LOGISTICREGRESSION_H
#define METIS_LOGISTICREGRESSION_H

#include "DataLabeled.h"

namespace metis {

    class LogisticRegression {

    private:

        // 3. Structure
        Eigen::MatrixXd _coeff;
        Eigen::VectorXd _intercept;

        unsigned _iterations = 10;
        double _learnRate = 1;
        unsigned _batchSize = 1;

        // 4. State
        unsigned _nAttributes;
        unsigned _nClasses;

        bool _verbose = false;

        // 5. Helper methods
        void batchGradientDescent(unsigned c);
        void stochasticGradientDescent(unsigned c);

    public:

        // 2. Interface methods
        double fit(DataLabeled *trainData, DataLabeled *testData, bool verbose);
        double fit(DataLabeled *trainData, DataLabeled *testData);
        double fit(DataLabeled *trainData);

        Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *data) const;
        Eigen::MatrixXd predictProbabilities(DataSet *data) const;

        Eigen::VectorXi predict(Eigen::MatrixXd *data) const;
        Eigen::VectorXi predict(DataSet *data) const;

        double score(DataLabeled *data) const;

        Eigen::MatrixXd getCoefficients();
        Eigen::VectorXd getIntercepts();

        // 1. Construction
        LogisticRegression(unsigned iterations, double learnRate, unsigned batchSize);
        LogisticRegression();
        ~LogisticRegression();

    };

}


#endif //METIS_LOGISTICREGRESSION_H
