//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_DECISIONTREE_H
#define METIS_DECISIONTREE_H

#include "DataLabeled.h"
#include "DTNode.h"

namespace metis {

class DecisionTree {

private:

    // 3. Structure
    DTNode *_root;

    // 4. State
    unsigned _nAttributes;
    unsigned _nOutcomes;
    std::vector<std::vector<std::string>> _categories;
    std::vector<std::string> _outcomes;

    // 5. Helper methods


public:

    // 2. Interface methods
    void fit(DataLabeled *data);

    Eigen::MatrixXd predictProbabilities(Eigen::MatrixXd *data) const;
    Eigen::MatrixXd predictProbabilities(DataSet *data) const;

    Eigen::VectorXi predict(Eigen::MatrixXd *data) const;
    Eigen::VectorXi predict(DataSet *data) const;

    double score(DataLabeled *data) const;

    void print() const;

    // 1. Construction
    DecisionTree();
    ~DecisionTree();

};

}


#endif //METIS_DECISIONTREE_H
