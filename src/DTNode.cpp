//
// Copyright (c) 2018 Andreas Bampouris
//

#include "../include/DTNode.h"

double metis::DTNode::measureInfoGain(metis::DataLabeled *data, std::vector<unsigned> subSet, unsigned attr, std::vector<std::vector<unsigned>> *subSets, Eigen::VectorXd *entropies) const {

    double gain = _entropy;

    Eigen::MatrixXd *in = data->getInputs()->getData();
    Eigen::MatrixXd *out = data->getOutputs()->getData();
    unsigned nCategories = data->categories(attr);
    unsigned nClasses = data->classes(0);
    unsigned nInstances = subSet.size();

    subSets->clear();
    subSets->resize(nCategories);
    for (unsigned i = 0; i < nInstances; ++i)
        subSets->at((unsigned)(in->coeff(subSet[i], attr))).push_back(subSet[i]);

    Eigen::VectorXd frequency(nClasses);
    entropies->resize(nCategories);
    entropies->setZero();

    for (unsigned c = 0; c < nCategories; ++c) {

        frequency.setZero();
        for (unsigned i = 0; i < subSets->at(c).size(); ++i)
            frequency.coeffRef((unsigned)(out->coeff(subSets->at(c)[i], 0)))++;
        frequency.array() /= subSets->at(c).size();

        for (unsigned o = 0; o < nClasses; ++o) {
            if (frequency.coeff(o) == 0) entropies->coeffRef(c) -= 0;
            else entropies->coeffRef(c) -= frequency.coeff(o) * log2(frequency.coeff(o));
        }

        gain -= ((double)(subSets->at(c).size()) / (double)(nInstances)) * entropies->coeff(c);

    }

    return gain;

}

unsigned metis::DTNode::findBestSplit(metis::DataLabeled *data, std::vector<unsigned> subSet, std::vector<unsigned> attributes, std::vector<std::vector<unsigned>> *bestSubSets, Eigen::VectorXd *bestEntropies) const {

    std::vector<std::vector<unsigned>> newSubSets;
    Eigen::VectorXd newEntropies;

    double infoGain = measureInfoGain(data, subSet, attributes[0], &newSubSets, &newEntropies);
    double bestGain = infoGain;
    unsigned bestSplit = attributes[0];
    *bestSubSets = newSubSets;
    *bestEntropies = newEntropies;

    for (unsigned a = 1; a < attributes.size(); ++a) {
        infoGain = measureInfoGain(data, subSet, attributes[a], &newSubSets, &newEntropies);
        if (infoGain >= bestGain) {
            bestGain = infoGain;
            bestSplit = attributes[a];
            *bestSubSets = newSubSets;
            *bestEntropies = newEntropies;
        }
    }

    return bestSplit;

}

void metis::DTNode::split(metis::DataLabeled *data, std::vector<unsigned> subSet, std::vector<unsigned> attributes) {

    if (!attributes.empty()) {

        std::vector<std::vector<unsigned>> newSubSets;
        Eigen::VectorXd newEntropies;
        _splitAttribute = findBestSplit(data, subSet, attributes, &newSubSets, &newEntropies);

        _nChildren = newSubSets.size();
        _children.resize(_nChildren);
        _splitValues.resize(_nChildren);

        Eigen::MatrixXd *out = data->getOutputs()->getData();
        Eigen::ArrayXd outcomeFrequencies(data->classes(0));
        unsigned nInstances;

        unsigned c;

//#pragma omp parallel for
        for (c = 0; c < _nChildren; ++c) {

            nInstances = newSubSets[c].size();
            outcomeFrequencies.setZero();

            for (unsigned i = 0; i < nInstances; ++i)
                ++outcomeFrequencies.coeffRef((unsigned)(out->coeff(newSubSets[c][i], 0)));
            outcomeFrequencies.array() /= nInstances;

            _children[c] = new DTNode(newEntropies.coeff(c), outcomeFrequencies);
            _splitValues.coeffRef(c) = c;

        }

        unsigned nNewAttributes = attributes.size() - 1;
        if (nNewAttributes > 0) {

            std::vector<unsigned> newAttributes;
            for (unsigned a = 0; a < nNewAttributes + 1; ++a)
                if (attributes[a] != _splitAttribute)
                    newAttributes.push_back(attributes[a]);

            for (unsigned c = 0; c < _nChildren; ++c)
                if (newEntropies.coeff(c) > 0)
                    _children[c]->split(data, newSubSets[c], newAttributes);

        }

    }

}

Eigen::VectorXd metis::DTNode::traverse(Eigen::VectorXd data) const {

    for (unsigned c = 0; c < _nChildren; ++c)
        if (data.coeff(_splitAttribute) == _splitValues.coeff(c))
            return _children[c]->traverse(data);

    return _outcomeFrequencies;

}

void metis::DTNode::printSplit(unsigned depth, unsigned attribute, double value, std::vector<std::vector<std::string>> categories, std::vector<std::string> outcomes) const {

    if (depth > 0) {
        for (unsigned d = 0; d < depth; ++d)
            std::cout << "--" << std::flush;
        std::cout << " " << attribute << " = " << categories[attribute].at((unsigned)(value)) << std::flush;
        std::cout << " [" << _entropy << "]" << std::flush;
        std::cout << " {"  << _outcomeFrequencies.transpose() << "}" << std::endl;
    }

    if (_children.size() > 0) {

        for (unsigned d = 0; d < depth; ++d)
            std::cout << "**" << std::flush;
        std::cout << "** " << _splitAttribute << std::endl;

        for (unsigned c = 0; c < _children.size(); ++c)
            _children[c]->printSplit(depth + 1, _splitAttribute, _splitValues[c], categories, outcomes);

    }

}

metis::DTNode::DTNode(double entropy, Eigen::VectorXd outcomeFrequencies) {

    _entropy = entropy;
    _outcomeFrequencies = outcomeFrequencies;

}

metis::DTNode::~DTNode() {

}
