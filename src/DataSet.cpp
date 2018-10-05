//
// Copyright (c) 2018 Andreas Bampouris
//

#include <fstream>
#include "../include/DataSet.h"

void metis::DataSet::create(std::vector<unsigned> attributes, std::vector<unsigned> categoricalAttr,
                            std::vector<std::map<std::string, unsigned>> dictionary) {

    _nAttributes = attributes.size();
    _nInstances = _nRows;

    _data.resize(_nInstances, _nAttributes);

    // Implementation details [correlating columns with attributes]
    bool colIsAttribute[_nColumns];
    for (unsigned c = 0; c < _nColumns; ++c) colIsAttribute[c] = false;
    for (unsigned a : attributes) colIsAttribute[a] = true;

    std::map<unsigned, unsigned> idOfColumn;
    for (unsigned a = 0; a < attributes.size(); ++a)
        idOfColumn.insert(std::pair<unsigned, unsigned>(attributes[a], a));

    // Implementation details [mapping categorical attributes to numbers]
    bool attrIsCategorical[_nColumns];
    for (unsigned c = 0; c < _nColumns; ++c) attrIsCategorical[c] = false;
    for (unsigned a : categoricalAttr) attrIsCategorical[a] = true;

    _nCategories.resize(_nAttributes);
    _categories.resize(_nAttributes);
    _idOfCategory.resize(_nAttributes);

    if (dictionary.size() != 0) _idOfCategory = dictionary;

    // File loading
    std::ifstream dataFile;
    dataFile.open(_filePath);

    unsigned row = 0;
    unsigned col = 0;
    unsigned attr;
    std::string rowData, colData;
    std::istringstream unscrambler;

    char *ch;

    while (getline(dataFile, rowData)) {

        col = 0;

        if (row < _nInstances) {

            unscrambler = std::istringstream(rowData);

            while (getline(unscrambler, colData, _separator)) {
                if (colIsAttribute[col] && colData[0] != 0) {

                    attr = idOfColumn[col];

                    if (!attrIsCategorical[col])
                        _data.coeffRef(row, attr) = strtod(colData.c_str(), &ch);

                        // if attribute is categorical, the string - number map
                        // is going to start being created
                    else {
                        if (std::find(_categories[attr].begin(), _categories[attr].end(),
                                      colData) == _categories[attr].end()) {
                            _idOfCategory[attr].insert(std::pair<std::string, unsigned>(colData,
                                                                                        _categories[attr].size()));
                            _categories[attr].push_back(colData);
                        }
                        _data.coeffRef(row, attr) = (double) _idOfCategory[attr][colData];
                    }

                }
                ++col;
            }

        } else break;

        ++row;

    }

    dataFile.close();

    for (unsigned a = 0; a < _nAttributes; ++a)
        _nCategories[a] = _categories[a].size();

}

void metis::DataSet::create(std::vector<unsigned> attributes,
                            std::vector<unsigned> categoricalAttr) {
    create(attributes, categoricalAttr, {});
}

void metis::DataSet::create(std::vector<unsigned> attributes) {
    create(attributes, {}, {});
}

void metis::DataSet::print() const {

    std::cout << _data << std::endl;

}

void metis::DataSet::producePlotInstructions(unsigned x, unsigned y) const {

    // Create data file
    std::ofstream fileData;
    fileData.open("points.txt");

    for (unsigned i = 0; i < _nInstances; ++i) {
        for (unsigned a = 0; a < _nAttributes; ++a) {
            fileData << _data.coeffRef(i, a) << " " << std::flush;
        }
        fileData << std::endl;
    }

    fileData.close();

    // Create instructions file
    std::ofstream fileInstr;
    fileInstr.open("instructions.txt");

    fileInstr << "plot \"points.txt\" u " << x + 1 << ":" << y + 1 << std::endl;

    fileInstr.close();

}

unsigned metis::DataSet::instances() const {
    return _nInstances;
}

unsigned metis::DataSet::attributes() const {
    return _nAttributes;
}

unsigned metis::DataSet::categories(unsigned attribute) const {
    return _nCategories[attribute];
}

std::vector<std::vector<std::string>> metis::DataSet::getCategories() const {
    return _categories;
}

std::vector<std::map<std::string, unsigned>> metis::DataSet::getDictionary() const {
    return _idOfCategory;
}

Eigen::MatrixXd *metis::DataSet::getData() {
    return &_data;
}

void metis::DataSet::shuffle(Eigen::VectorXi order) {
    _data = order.asPermutation() * _data;
}

void metis::DataSet::shrink(unsigned from, unsigned to) {

    _nInstances = to - from;
    Eigen::MatrixXd newData(_nInstances, _nAttributes);

    unsigned cur = 0;
    for (unsigned r = from; r < to; ++r)
        newData.row(cur++) = _data.row(r);

    _data = newData;

}

void metis::DataSet::convertToBinaryAttributes() {

    if (_nAttributes > 1) {
        std::cout << "This conversion can only be applied to single binary "
                     "attribute datasets." << std::endl;
        exit(1);
    }

    if (_nCategories[0] == 0) {
        std::cout << "This conversion can only be applied to categorical "
                     "attribute datasets." << std::endl;
        exit(1);
    }

    double binLookup[_nCategories[0]][_nCategories[0]];
    for (unsigned i = 0; i < _nCategories[0]; ++i) {
        for (unsigned j = 0; j < _nCategories[0]; ++j) {
            binLookup[i][j] = 0.0;
        }
        binLookup[i][i] = 1.0;
    }

    unsigned exampleClass;
    Eigen::MatrixXd newData(_nInstances, _nCategories[0]);
    for (unsigned i = 0; i < _nInstances; ++i) {
        for (unsigned c = 0; c < _nCategories[0]; ++c) {
            exampleClass = (unsigned) (_data.coeffRef(i, 0) + 0.5);
            newData.coeffRef(i, c) = binLookup[exampleClass][c];
        }
    }

    _data = newData;

    _nAttributes = _nCategories[0];

    _nCategories.clear();

    _nCategories.resize(_nAttributes);
    for (unsigned a = 0; a < _nAttributes; ++a) _nCategories[a] = 2;

    std::vector<std::vector<std::string>> newCategories(_nAttributes);
    std::vector<std::map<std::string, unsigned>> newIdOfCategory(_nAttributes);

    for (unsigned a = 0; a < _nAttributes; ++a) {
        newCategories[a].resize(2);
        newCategories[a][0] = "not ";
        newCategories[a][0].append(_categories[0][a]);
        newCategories[a][1] = _categories[0][a];

        newIdOfCategory[a].insert(std::pair<std::string, unsigned>(newCategories[a][0], 0));
        newIdOfCategory[a].insert(std::pair<std::string, unsigned>(newCategories[a][1], 1));
    }

    _categories = newCategories;
    _idOfCategory = newIdOfCategory;

}

void metis::DataSet::applyStandardization() {

    _means.resize(_nAttributes);
    _stDev.resize(_nAttributes);

    _means.array() = _data.array().colwise().mean();

    for (unsigned a = 0; a < _nAttributes; ++a) {
        _stDev.coeffRef(a) = std::sqrt((_data.col(a).array() - _means.coeffRef(a)).square().sum() / (_nInstances - 1));
        if (_stDev.coeffRef(a) == 0) _stDev.coeffRef(a) = 1;
    }

    _data.array().rowwise() -= _means.transpose().array();
    _data.array().rowwise() /= _stDev.transpose().array();

}

void metis::DataSet::applyLogTransform() {
    _data.array() = _data.array().log();
}

void metis::DataSet::applyExpTransform() {
    _data.array() = _data.array().exp();
}

void metis::DataSet::scaleSquaredLengthToOne() {

    double scale = std::sqrt(_data.array().pow(2).sum() / _nAttributes);
    _data.array() /= scale;

}

void metis::DataSet::readThrough() {

    std::ifstream dataFile;
    dataFile.open(_filePath);

    if (!dataFile) {
        std::cerr << "Unable to open file." << std::endl;
        exit(1);
    }

    // Read one row and see how many columns it has on assumption that all
    // rows have the same number of columns. Then count number of all rows.
    std::string rowData, columnData;

    getline(dataFile, rowData);
    _nRows = 1;
    _nColumns = 0;

    std::istringstream unscrambler(rowData);
    while (getline(unscrambler, columnData, _separator))
        if (columnData[0] != 0) ++_nColumns;

    while (getline(dataFile, rowData)) ++_nRows;

    dataFile.close();

}

metis::DataSet::DataSet(std::string &filePath, char separator) {

    _filePath = filePath;
    _separator = separator;
    readThrough();

}

metis::DataSet::DataSet(Eigen::MatrixXd data) {

    _data = data;
    _nAttributes = _data.cols();
    _nInstances = _data.rows();

}

metis::DataSet::DataSet() {

}

metis::DataSet::~DataSet() {

}
