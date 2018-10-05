//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_DTNODE_H
#define METIS_DTNODE_H

#include "DataLabeled.h"

namespace metis {

class DTNode {

private:

    // 3. Structure
    std::vector<DTNode *> _children;
    unsigned _splitAttribute;
    Eigen::VectorXd _splitValues;
    unsigned _nChildren = 0;

    // 4. State
    double _entropy;
    Eigen::VectorXd _outcomeFrequencies;

    // 5. Helper methods
    double measureInfoGain(metis::DataLabeled *data, std::vector<unsigned> subSet, unsigned attr, std::vector<std::vector<unsigned>> *subSets, Eigen::VectorXd *entropies) const;
    unsigned findBestSplit(metis::DataLabeled *data, std::vector<unsigned> subSet, std::vector<unsigned> attributes, std::vector<std::vector<unsigned>> *bestSubSets, Eigen::VectorXd *bestEntropies) const;

public:

    // 2. Interface methods
    void split(metis::DataLabeled *data, std::vector<unsigned> subSet, std::vector<unsigned> attributes);
    Eigen::VectorXd traverse(Eigen::VectorXd data) const;
    void printSplit(unsigned depth, unsigned attribute, double value, std::vector<std::vector<std::string>> categories, std::vector<std::string> outcomes) const;

    // 1. Construction
    DTNode(double entropy, Eigen::VectorXd outcomeFrequencies);
    ~DTNode();

};

}


#endif //METIS_DTNODE_H
