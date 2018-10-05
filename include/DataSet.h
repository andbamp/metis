//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_DATASET_H
#define METIS_DATASET_H


#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <map>

namespace metis {

class DataSet {

private:

    // 3. Structure
    Eigen::MatrixXd _data;

    std::vector<std::vector<std::string>> _categories;
    std::vector<std::map<std::string, unsigned>> _idOfCategory;

    Eigen::VectorXd _means;
    Eigen::VectorXd _stDev;

    // 4. State
    unsigned _nInstances = 0;
    unsigned _nAttributes = 0;
    std::vector<unsigned> _nCategories;

    std::string _filePath;
    char _separator;
    unsigned _nRows = 0;
    unsigned _nColumns = 0;

    // 5. Helper methods
    void readThrough();


public:

    // 2. Interface methods
    void create(std::vector<unsigned> attributes, std::vector<unsigned> categoricalAttr,
                std::vector<std::map<std::string, unsigned>> dictionary);
    void create(std::vector<unsigned> attributes, std::vector<unsigned> categoricalAttr);
    void create(std::vector<unsigned> attributes);

    void print() const;
    void producePlotInstructions(unsigned x, unsigned y) const;

    // b. Getters
    unsigned instances() const;
    unsigned attributes() const;
    unsigned categories(unsigned attribute) const;

    std::vector<std::vector<std::string>> getCategories() const;
    std::vector<std::map<std::string, unsigned>> getDictionary() const;

    Eigen::MatrixXd *getData();

    // c. Transformations
    void shuffle(Eigen::VectorXi order);
    void shrink(unsigned from, unsigned to);

    void convertToBinaryAttributes();
    void applyStandardization();
    void applyLogTransform();
    void applyExpTransform();
    void scaleSquaredLengthToOne();

    // 1. Construction
    DataSet(std::string &filePath, char separator);
    DataSet(Eigen::MatrixXd data);
    DataSet();
    ~DataSet();

};

}


#endif //METIS_DATASET_H
