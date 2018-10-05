//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/DecisionTree.h"

void metis::DecisionTree::fit(metis::DataLabeled *data) {

    _nAttributes = data->inputs();
    _nOutcomes = data->classes(0);
    _categories = data->getInputs()->getCategories();
    _outcomes = data->getOutputs()->getCategories().at(0);

    Eigen::MatrixXd *out = data->getOutputs()->getData();
    Eigen::ArrayXd outcomeFrequencies(_nOutcomes);
    unsigned nInstances = data->instances();

    outcomeFrequencies.setZero();
    for (unsigned i = 0; i < nInstances; ++i)
        ++outcomeFrequencies.coeffRef((unsigned)(out->coeff(i, 0)));
    outcomeFrequencies.array() /= nInstances;

    _root = new DTNode(1, outcomeFrequencies);

    std::vector<unsigned> subSet(nInstances);
    for (unsigned i = 0; i < data->instances(); ++i)
        subSet[i] = i;

    std::vector<unsigned> attributes(_nAttributes);
    for (unsigned a = 0; a < _nAttributes; ++a)
        attributes[a] = a;

    _root->split(data, subSet, attributes);

}

Eigen::VectorXi metis::DecisionTree::predict(Eigen::MatrixXd *data) const {

    unsigned nInstances = data->rows();

    Eigen::MatrixXd probabilities(nInstances, _nOutcomes);
    probabilities = predictProbabilities(data);

    Eigen::VectorXi prediction(nInstances);
    double predictedProbability;

    for (unsigned i = 0; i < nInstances; ++i) {
        prediction.coeffRef(i) = 0;
        predictedProbability = probabilities.coeff(i, 0);
        for (unsigned c = 1; c < _nOutcomes; ++c) {
            if (probabilities.coeff(i, c) > predictedProbability) {
                prediction.coeffRef(i) = c;
                predictedProbability = probabilities.coeff(i, c);
            }
        }
    }

    return prediction;

}

Eigen::VectorXi metis::DecisionTree::predict(metis::DataSet *data) const {
    return predict(data->getData());
}

Eigen::MatrixXd metis::DecisionTree::predictProbabilities(Eigen::MatrixXd *data) const {

    Eigen::MatrixXd predictions(data->rows(), _nOutcomes);

    for (unsigned i = 0; i < data->rows(); ++i)
        predictions.row(i) = _root->traverse(data->row(i)).transpose();

    return predictions;

}

Eigen::MatrixXd metis::DecisionTree::predictProbabilities(metis::DataSet *data) const {
    return predictProbabilities(data->getData());
}

double metis::DecisionTree::score(metis::DataLabeled *data) const {

    Eigen::VectorXi predictions(data->instances());
    predictions = predict(data->getInputs()->getData());

    unsigned correct = 0;
    for (unsigned i = 0; i < data->instances(); ++i)
        if (data->getOutputs()->getData()->coeff(i) == predictions.coeff(i))
            ++correct;

    return (double)correct / (double)data->instances();
}

void metis::DecisionTree::print() const {
    _root->printSplit(0, 0, 0, _categories, _outcomes);
}

metis::DecisionTree::DecisionTree() {

}

metis::DecisionTree::~DecisionTree() {

}
