//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_MLPCLASSIFIER_H
#define METIS_MLPCLASSIFIER_H

#include "DataLabeled.h"

namespace metis {

class MLPClassifier {

private:

    // 3. Structure
    std::vector<Eigen::MatrixXd> _coeff;
    std::vector<Eigen::VectorXd> _intercept;

    std::vector<unsigned> _topology;
    std::vector<unsigned> _activation;

    unsigned _batchSize = 1;
    double _learnRate = 1;
    unsigned _iterations = 1;

    // 4. State
    unsigned _nLayers;
    unsigned _nAttributes;
    unsigned _nClasses;

    bool _verbose;

    // 5. Helper functions
    void activationSigmoid(Eigen::MatrixXd *mat) const;
    void activationReLU(Eigen::MatrixXd *mat) const;
    void activationFunction(Eigen::MatrixXd *activated, unsigned actFunction) const;

    void derivativeSigmoid(Eigen::MatrixXd *mat) const;
    void derivativeReLU(Eigen::MatrixXd *mat) const;
    void derivativeFunction(Eigen::MatrixXd *linear, Eigen::MatrixXd *activated, unsigned actFunction) const;

    void batchGradientDescent();
    void stochasticGradientDescent();

    void combineWeights();

public:

    // 2. Interface methods
    double train(DataLabeled *trainData, DataLabeled *testData, bool verbose);
    double train(DataLabeled *trainData, DataLabeled *testData);
    double train(DataLabeled *trainData);

    Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *data) const;
    Eigen::MatrixXd predictProbabilities(DataSet *data) const;

    Eigen::VectorXi predict(Eigen::MatrixXd *data) const;
    Eigen::VectorXi predict(DataSet *data) const;

    double score(DataLabeled *data) const;

    // 1. Construction
    MLPClassifier(std::vector<unsigned> hiddenLayers, std::vector<unsigned> activationFunction, unsigned batchSize, double learnRate, unsigned iterations);
    MLPClassifier();
    ~MLPClassifier();

};

}


#endif //METIS_MLPCLASSIFIER_H
