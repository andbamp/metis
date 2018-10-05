//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/MLPClassifier.h"

namespace {

    Eigen::MatrixXd *g_in;
    Eigen::MatrixXd *g_out;
    unsigned g_nInstances;
    unsigned g_nBatches;

    std::vector<Eigen::MatrixXd> *g_coeff;
    std::vector<Eigen::VectorXd> *g_intercept;
    int g_nThreads;
    unsigned g_counter = 0;
    unsigned *g_counter_n;

    metis::DataLabeled *trainD;
    metis::DataLabeled *testD;

}

void metis::MLPClassifier::activationReLU(Eigen::MatrixXd *mat) const {

//    mat->array() = mat->array().exp();
//    mat->array() += 1.0;
//    mat->array() = mat->array().log();

    for (unsigned r = 0; r < mat->rows(); ++r) {
        for (unsigned c = 0; c < mat->cols(); ++c) {
            if (mat->coeffRef(r, c) <= 0) mat->coeffRef(r, c) = 0;
        }
    }

}

void metis::MLPClassifier::activationSigmoid(Eigen::MatrixXd *mat) const {

    mat->array() *= -1;
    mat->array() = mat->array().exp();
    mat->array() += 1.0;
    mat->array() = mat->array().inverse();

}

void metis::MLPClassifier::activationFunction(Eigen::MatrixXd *activated, unsigned actFunction) const {

    switch (actFunction) {
        case 0:
            activationSigmoid(activated);
            break;
        case 1:
            activationReLU(activated);
            break;
        default:
            activationSigmoid(activated);
            break;
    }

}

void metis::MLPClassifier::derivativeSigmoid(Eigen::MatrixXd *mat) const {
    mat->array() *= (1.0 - mat->array());
}

void metis::MLPClassifier::derivativeReLU(Eigen::MatrixXd *mat) const {

//    mat->array() *= -1;
//    mat->array() = mat->array().exp();
//    mat->array() += 1.0;
//    mat->array() = mat->array().inverse();

    for (unsigned r = 0; r < mat->rows(); ++r) {
        for (unsigned c = 0; c < mat->cols(); ++c) {
            if (mat->coeffRef(r, c) <= 0) mat->coeffRef(r, c) = 0;
            else mat->coeffRef(r, c) = 1;
        }
    }

}

void metis::MLPClassifier::derivativeFunction(Eigen::MatrixXd *linear, Eigen::MatrixXd *activated,
                                                unsigned actFunction) const {

    switch (actFunction) {
        case 0:
            derivativeSigmoid(activated);
            break;
        case 1:
            activated->array() = linear->array();
            derivativeReLU(activated);
            break;
        default:
            derivativeSigmoid(activated);
            break;
    }

}

void metis::MLPClassifier::combineWeights() {

    _coeff = g_coeff[0];
    _intercept = g_intercept[0];
    for (unsigned th = 1; th < g_nThreads; ++th) {
        for (unsigned l = 0; l < _nLayers; ++l) {
            _coeff[l] += g_coeff[th][l];
            _intercept[l] += g_intercept[th][l];
        }
    }
    for (unsigned l = 0; l < _nLayers; ++l) {
        _coeff[l] /= g_nThreads;
        _intercept[l] /= g_nThreads;
    }

}

void metis::MLPClassifier::batchGradientDescent() {

    unsigned th = omp_get_thread_num();

    g_counter_n[th] = 0;

    Eigen::MatrixXd linear[_nLayers];
    Eigen::MatrixXd activated[_nLayers];
    Eigen::MatrixXd errors[_nLayers];

    Eigen::MatrixXd dw[_nLayers];
    Eigen::VectorXd db[_nLayers];

    for (unsigned l = 0; l < _nLayers; ++l) {
        linear[l].resize(_topology[l+1], g_nBatches);
        activated[l].resize(_topology[l+1], g_nBatches);
        errors[l].resize(_topology[l+1], g_nBatches);

        dw[l].resize(_topology[l+1], _topology[l]);
        db[l].resize(_topology[l+1]);
    }

    for (unsigned i = 0; i < _iterations; ++i) {

        for (unsigned b = th; b < g_nBatches; b += g_nThreads) {

            // Feed-forward
            linear[0] = g_coeff[th][0] * g_in->block(b * _batchSize, 0, _batchSize, _nAttributes).transpose();
            linear[0].colwise() += g_intercept[th][0];
            activated[0] = linear[0];
            activationFunction(&activated[0], _activation[0]);

            for (unsigned l = 1; l < _nLayers; ++l) {
                linear[l] = g_coeff[th][l] * activated[l-1];
                linear[l].colwise() += g_intercept[th][l];
                activated[l] = linear[l];
                activationFunction(&activated[l], _activation[l]);
            }

            // Back-propagation
            errors[_nLayers - 1] = activated[_nLayers - 1] - g_out->block(b * _batchSize, 0, _batchSize, _nClasses).transpose();
            dw[_nLayers - 1] = errors[_nLayers - 1] * activated[_nLayers - 2].transpose();
            db[_nLayers - 1] = errors[_nLayers - 1].colwise().sum();

            for (int l = _nLayers - 2; l > 0; --l) {
                errors[l] = g_coeff[th][l+1].transpose() * errors[l+1];
                derivativeFunction(&linear[l], &activated[l], _activation[l]);
                errors[l].array() *= activated[l].array();
                dw[l] = errors[l] * activated[l-1].transpose();
                db[l] = errors[l].rowwise().sum();
            }

            errors[0] = g_coeff[th][1].transpose() * errors[1];
            derivativeFunction(&linear[0], &activated[0], _activation[0]);
            errors[0].array() *= activated[0].array();
            dw[0] = errors[0] * g_in->block(b * _batchSize, 0, _batchSize, _nAttributes);
            db[0] = errors[0].rowwise().sum();

            // Update weights
            for (unsigned l = 0; l < _nLayers - 1; ++l) {
                g_coeff[th][l] -= _learnRate * (dw[l] / _batchSize);
                g_intercept[th][l] -= _learnRate * (db[l] / _batchSize);
            }

            if (g_counter > 0 && (g_counter % 50) == 0) {
#pragma omp barrier
                if (th == 0) {
                    if (_verbose)
                        std::cout << "Score on test set after " << g_counter << " updates:\t" << score(testD) << std::endl;
                    combineWeights();
                }
#pragma omp barrier
                for (unsigned l = 0; l < _nLayers - 1; ++l) {
                    g_coeff[th][l] = _coeff[l];
                    g_intercept[th][l] = _intercept[l];
                }
            }

            ++g_counter_n[th];
            ++g_counter;

        }

    }

}

void metis::MLPClassifier::stochasticGradientDescent() {

    Eigen::MatrixXd linear[_nLayers];
    Eigen::MatrixXd activated[_nLayers];
    Eigen::MatrixXd errors[_nLayers];

    Eigen::MatrixXd dw[_nLayers];
    Eigen::VectorXd db[_nLayers];

    for (unsigned l = 0; l < _nLayers; ++l) {
        linear[l].resize(_topology[l+1], 1);
        activated[l].resize(_topology[l+1], 1);
        errors[l].resize(_topology[l+1], 1);

        dw[l].resize(_topology[l+1], _topology[l]);
        db[l].resize(_topology[l+1]);
    }

    for (unsigned i = 0; i < _iterations; ++i) {

        for (unsigned r = 0; r < g_nInstances; ++r) {

            // Feed-forward
            linear[0] = _coeff[0] * g_in->row(r).transpose();
            linear[0].colwise() += _intercept[0];
            activated[0] = linear[0];
            activationFunction(&activated[0], _activation[0]);

            for (unsigned l = 1; l < _nLayers; ++l) {
                linear[l] = _coeff[l] * activated[l-1];
                linear[l].colwise() += _intercept[l];
                activated[l] = linear[l];
                activationFunction(&activated[l], _activation[l]);
            }

            // Back-propagation
            errors[_nLayers - 1] = activated[_nLayers - 1] - g_out->row(r).transpose();
            dw[_nLayers - 1] = errors[_nLayers - 1] * activated[_nLayers - 2].transpose();
            db[_nLayers - 1] = errors[_nLayers - 1];

            for (int l = _nLayers - 2; l > 0; --l) {
                errors[l] = _coeff[l+1].transpose() * errors[l+1];
                derivativeFunction(&linear[l], &activated[l], _activation[l]);
                errors[l].array() *= activated[l].array();
                dw[l] = errors[l] * activated[l-1].transpose();
                db[l] = errors[l];
            }

            errors[0] = _coeff[1].transpose() * errors[1];
            derivativeFunction(&linear[0], &activated[0], _activation[0]);
            errors[0].array() *= activated[0].array();
            dw[0] = errors[0] * g_in->row(r);
            db[0] = errors[0];

            // Update weights
            for (unsigned l = 0; l < _nLayers - 1; ++l) {
                _coeff[l] -= _learnRate * dw[l];
                _intercept[l] -= _learnRate * db[l];
            }

        }

    }

}

double metis::MLPClassifier::train(metis::DataLabeled *trainData, metis::DataLabeled *testData, bool verbose) {

    rand();

    _nAttributes = trainData->inputs();
    _nClasses = trainData->outputs();

    _verbose = verbose;

    g_in = trainData->getInputs()->getData();
    g_out = trainData->getOutputs()->getData();

    g_nInstances = trainData->instances();
    if (_batchSize == 0) _batchSize = g_nInstances;
    g_nBatches = g_nInstances / _batchSize;

    DataSet outData;
    if (_nClasses == 1) {
        outData = *(trainData->getOutputs());
        outData.convertToBinaryAttributes();
        _nClasses = outData.attributes();
        g_out = outData.getData();
    }

    _topology[0] = _nAttributes;
    _topology[_nLayers] = _nClasses;

    _coeff.resize(_nLayers);
    _intercept.resize(_nLayers);
    for (unsigned l = 0; l < _nLayers; ++l) {
        _coeff[l].resize(_topology[l + 1], _topology[l]);
        _intercept[l].resize(_topology[l + 1]);
    }

#pragma omp parallel
    {
        g_nThreads = omp_get_num_threads();
    };

    g_coeff = new std::vector<Eigen::MatrixXd>[g_nThreads];
    g_intercept = new std::vector<Eigen::VectorXd>[g_nThreads];

#pragma omp parallel
    {
        unsigned th = omp_get_thread_num();

        g_coeff[th].resize(_nLayers);
        g_intercept[th].resize(_nLayers);
        for (unsigned l = 0; l < _nLayers; ++l) {
            g_coeff[th][l].resize(_topology[l + 1], _topology[l]);
            g_coeff[th][l].setRandom();
            g_intercept[th][l].resize(_topology[l + 1]);
            g_intercept[th][l].setRandom();
        }
    };

    double bestScore;

    trainD = trainData;
    testD = testData;

    if (_batchSize == 1) stochasticGradientDescent();

    else {

        g_counter_n = new unsigned[g_nThreads];

        g_counter = 0;
#pragma omp parallel reduction(+:g_counter)
        {
            batchGradientDescent();
        };

    }

    combineWeights();

    bestScore = score(testData);

    return bestScore;

}

double metis::MLPClassifier::train(metis::DataLabeled *trainData, metis::DataLabeled *testData) {
    return train(trainData, trainData, false);
}

double metis::MLPClassifier::train(metis::DataLabeled *trainData) {
    return train(trainData, trainData);
}

Eigen::MatrixXd metis::MLPClassifier::predictProbabilities(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();

    Eigen::MatrixXd probabilities[_nLayers];
    for (unsigned l = 0; l < _nLayers; ++l)
        probabilities[l].resize(_topology[l + 1], nInstances);

    probabilities[0] = _coeff[0] * data->transpose();
    probabilities[0].colwise() += _intercept[0];
    activationFunction(&probabilities[0], _activation[0]);

    for (unsigned l = 1; l < _nLayers; ++l) {
        probabilities[l] = _coeff[l] * probabilities[l-1];
        probabilities[l].colwise() += _intercept[l];
        activationFunction(&probabilities[l], _activation[l]);
    }

    return probabilities[_nLayers - 1];

}

Eigen::MatrixXd metis::MLPClassifier::predictProbabilities(metis::DataSet *data) const {
    return predictProbabilities(data->getData());
}

Eigen::VectorXi metis::MLPClassifier::predict(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();

    Eigen::MatrixXd probabilities(_nClasses, nInstances);
    probabilities = predictProbabilities(data);

    Eigen::VectorXi prediction(nInstances);
    double predictedLikelihood;

    for (unsigned i = 0; i < nInstances; ++i) {
        prediction.coeffRef(i) = 0;
        predictedLikelihood = probabilities.coeff(0, i);
        for (unsigned c = 1; c < _nClasses; ++c) {
            if (probabilities.coeff(c, i) > predictedLikelihood) {
                prediction.coeffRef(i) = c;
                predictedLikelihood = probabilities.coeff(c, i);
            }
        }
    }

    return prediction;

}

Eigen::VectorXi metis::MLPClassifier::predict(metis::DataSet *data) const {
    return predict(data->getData());
}

double metis::MLPClassifier::score(metis::DataLabeled *data) const {

    unsigned nInstances = data->instances();

    Eigen::VectorXi predictions(nInstances);
    predictions = predict(data->getInputs()->getData());

    unsigned correct = 0;
    for (unsigned i = 0; i < nInstances; ++i)
        if (data->getOutputs()->getData()->coeff(i) == predictions.coeff(i))
            ++correct;

    return (double)correct / (double)nInstances;

}

metis::MLPClassifier::MLPClassifier(std::vector<unsigned> hiddenLayers, std::vector<unsigned> activationFunction,
                                        unsigned batchSize, double learnRate, unsigned iterations) {

    _nLayers = hiddenLayers.size() + 1;
    _topology.resize(_nLayers + 1);
    _activation.resize(_nLayers);

    for (unsigned l = 1; l < _nLayers; ++l)
        _topology[l] = hiddenLayers[l - 1];

    for (unsigned l = 0; l < _nLayers; ++l)
        _activation[l] = activationFunction[l];

    _batchSize = batchSize;
    _learnRate = learnRate;
    _iterations = iterations;

}

metis::MLPClassifier::MLPClassifier() {

}

metis::MLPClassifier::~MLPClassifier() {

}
