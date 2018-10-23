//
// Copyright (c) 2018 Andreas Bampouris
//

#include <fstream>
#include <iostream>
#include "../include/DataContainer.h"

Eigen::MatrixXd metis::DataContainer::createNumericalMatrix(std::vector<unsigned> columns) {
    
    Eigen::MatrixXd data(_nInstances, columns.size());
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        if (_colIsAttribute[columns[c]]) {
            data.col(c) = _data.col(_c2a[columns[c]]);
        } else if (_colIsCatAttribute[columns[c]]) {
            data.col(c) = _catData.col(_c2ca[columns[c]]).cast<double>();
        } else {
            std::cerr << "Data for column " << c << " have not been loaded." << std::endl;
            exit(1);
        }
    }
    
    return data;
    
}

Eigen::MatrixXi metis::DataContainer::createCategoricalMatrix(std::vector<unsigned> columns) {
    
    Eigen::MatrixXi data(_nInstances, columns.size());
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        if (_colIsCatAttribute[columns[c]]) {
            data.col(c) = _catData.col(_c2ca[columns[c]]);
        } else if (_colIsAttribute[columns[c]]) {
            data.col(c) = _data.col(_c2a[columns[c]]).cast<int>();
        } else {
            std::cerr << "Data for column " << c << " have not been loaded." << std::endl;
            exit(1);
        }
    }
    
    return data;
    
}

Eigen::ArrayXi metis::DataContainer::createClassArray(unsigned column) {
    
    Eigen::ArrayXi data(_nInstances);
    
    if (_colIsCatAttribute[column]) {
        data = _catData.col(_c2ca[column]);
    } else if (_colIsAttribute[column]) {
        data = _data.col(_c2a[column]).cast<int>();
    } else {
        std::cerr << "Data for column " << column << " have not been loaded." << std::endl;
        exit(1);
    }
    
    return data;
    
}

Eigen::MatrixXd metis::DataContainer::createBinaryMatrix(unsigned column) {
    
    if (_colIsAttribute[column]) {
        std::cerr << "Cannot create binary matrix for numerical column " << column << "." << std::endl;
        exit(1);
    } else if (!_colIsCatAttribute[column]) {
        std::cerr << "Data for column " << column << " have not been loaded." << std::endl;
        exit(1);
    }
    
    unsigned nCategories = _nCategories[_c2ca[column]];
    
    // If column and row are the same, 1.0. Otherwise 0.0.
    double binLookup[nCategories][nCategories];
    for (unsigned i = 0; i < nCategories; ++i) {
        for (unsigned j = 0; j < nCategories; ++j) {
            binLookup[i][j] = 0.0;
        }
        binLookup[i][i] = 1.0;
    }
    
    Eigen::MatrixXd newData(_nInstances, nCategories);
    for (unsigned i = 0; i < _nInstances; ++i) {
        for (unsigned c = 0; c < nCategories; ++c) {
            newData.coeffRef(i, c) = binLookup[_catData.coeff(i, _c2ca[column])][c];
        }
    }
    
    return newData;
    
}

Eigen::MatrixXd metis::DataContainer::convertToBinaryMatrix(Eigen::ArrayXi *data) {
    
    unsigned nInstances = data->rows();
    
    unsigned nCategories = 0;
    for (unsigned i = 0; i < nInstances; ++i) {
        if (data->coeff(i) >= nCategories) {
            nCategories = data->coeff(i) + 1;
        }
    }
    
    double binLookup[nCategories][nCategories];
    for (unsigned i = 0; i < nCategories; ++i) {
        for (unsigned j = 0; j < nCategories; ++j) {
            binLookup[i][j] = 0.0;
        }
        binLookup[i][i] = 1.0;
    }
    
    Eigen::MatrixXd newData(nInstances, nCategories);
    for (unsigned i = 0; i < nInstances; ++i) {
        for (unsigned c = 0; c < nCategories; ++c) {
            newData.coeffRef(i, c) = binLookup[data->coeff(i)][c];
        }
    }
    
    return newData;
    
}

void metis::DataContainer::shuffle(Eigen::VectorXi order) {
    
    _data.matrix() = order.asPermutation() * _data.matrix();
    _catData.matrix() = order.asPermutation() * _catData.matrix();
    
}

void metis::DataContainer::shuffle() {
    
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(_nInstances, 0, _nInstances);
    std::random_shuffle(indices.data(), indices.data() + _nInstances);
    
    shuffle(indices);
    
}

void metis::DataContainer::shrink(unsigned from, unsigned to) {
    
    _nInstances = to - from;
    Eigen::ArrayXXd newData(_nInstances, _nAttributes);
    Eigen::ArrayXXi newCatData(_nInstances, _nCatAttributes);
    
    unsigned cur = 0;
    for (unsigned r = from; r < to; ++r) {
        newData.row(cur) = _data.row(r);
        newCatData.row(cur++) = _catData.row(r);
    }
    
    _data = newData;
    _catData = newCatData;
    
}

metis::DataContainer metis::DataContainer::split(double proportion) {
    
    double cut = _nInstances * (1 - proportion) + .5;
    unsigned newSize = (unsigned) cut;
    
    DataContainer splitOutput = *this;
    splitOutput.shrink(newSize, _nInstances);
    
    shrink(0, newSize);
    
    return splitOutput;
    
}

void metis::DataContainer::standardize(std::vector<unsigned> columns) {
    
    Eigen::ArrayXd mean(_nAttributes);
    Eigen::ArrayXd std(_nAttributes);
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        
        if (_colIsAttribute[columns[c]]) {
            
            // Standardization
            mean.coeffRef(c) = _data.col(_c2a[columns[c]]).mean();
            std.coeffRef(c) = std::sqrt(
                    (_data.col(_c2a[columns[c]]).array() - mean.coeff(c)).square().sum() / (_nInstances - 1) );
            if (std.coeff(c) == 0) std.coeffRef(c) = 1;
            
            _data.col(_c2a[columns[c]]) -= mean.coeffRef(c);
            _data.col(_c2a[columns[c]]) /= std.coeffRef(c);
            
        } else if (_colIsCatAttribute[columns[c]]) {
            std::cerr << "Cannot standardize categorical data in column " << columns[c] << "." << std::endl;
            exit(1);
        } else {
            std::cerr << "Data for column " << columns[c] << " have not been loaded." << std::endl;
            exit(1);
        }
        
    }
    
}

void metis::DataContainer::rescale(std::vector<unsigned> columns) {
    
    double scale = 0;
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        
        if (_colIsAttribute[columns[c]]) {
            
            scale += _data.col(columns[c]).pow(2).sum();
            
        } else if (_colIsCatAttribute[columns[c]]) {
            std::cerr << "Cannot rescale categorical data in column " << columns[c] << "." << std::endl;
            exit(1);
        } else {
            std::cerr << "Data for column " << columns[c] << " have not been loaded." << std::endl;
            exit(1);
        }
        
    }
    
    // Data are scaled to unit length.
    scale = std::sqrt(scale / columns.size());
    
    for (unsigned c = 0; c < columns.size(); ++c) _data.col(columns[c]) /= scale;
    
}

void metis::DataContainer::logTransform(std::vector<unsigned> columns) {
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        
        if (_colIsAttribute[columns[c]]) {
            
            _data.col(_c2a[columns[c]]) = _data.col(_c2a[columns[c]]).log();
            
        } else if (_colIsCatAttribute[columns[c]]) {
            std::cerr << "Cannot log transform categorical data in column " << columns[c] << "." << std::endl;
            exit(1);
        } else {
            std::cerr << "Data for column " << columns[c] << " have not been loaded." << std::endl;
            exit(1);
        }
        
    }
    
}

void metis::DataContainer::expTransform(std::vector<unsigned> columns) {
    
    for (unsigned c = 0; c < columns.size(); ++c) {
        
        if (_colIsAttribute[columns[c]]) {
            
            _data.col(_c2a[columns[c]]) = _data.col(_c2a[columns[c]]).exp();
            
        } else if (_colIsCatAttribute[columns[c]]) {
            std::cerr << "Cannot exp transform categorical data in column " << columns[c] << "." << std::endl;
            exit(1);
        } else {
            std::cerr << "Data for column " << columns[c] << " have not been loaded." << std::endl;
            exit(1);
        }
        
    }
    
}

void metis::DataContainer::print() const {
    
    for (unsigned r = 0; r < _nInstances; ++r) {
        for (unsigned c = 0; c < _nAttributes; ++c)
            std::cout << "\t" << _data.coeff(r, c) << std::flush;
        for (unsigned c = 0; c < _nCatAttributes; ++c)
            std::cout << "\t" << _catItoS[c][_catData.coeff(r, c)] << std::flush;
        std::cout << std::endl;
    }
    
}

metis::DataContainer::DataContainer(Eigen::ArrayXd data, Eigen::ArrayXi catData,
                              std::vector<std::vector<std::string>> catItoS,
                              std::vector<std::map<std::string, int>> catStoI) {
    
    _data = data;
    _catData = catData;
    
    _nInstances = data.rows();
    _nAttributes = _data.cols();
    _nCatAttributes = _catData.cols();
    _catItoS = catItoS;
    _catStoI = catStoI;
    
}

void metis::DataContainer::readFile(std::string &filePath, char separatorChar, std::string &missingValue,
                                 std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                                 unsigned nRows, unsigned nCols) {
    
    // File loading
    std::ifstream dataFile;
    dataFile.open(filePath);
    
    unsigned row = 0;
    unsigned col;
    unsigned attr;
    std::string rowData, colData;
    std::istringstream unscrambler;
    
    char *ch;
    
    while (getline(dataFile, rowData)) {
        
        col = 0;
        
        if (row < _nInstances) {
            
            unscrambler = std::istringstream(rowData);
            
            // Reads a line and distributes values of relevant columns between _data and _catData.
            while (getline(unscrambler, colData, separatorChar)) {
                
                if (_colIsAttribute[col] && colData[0] != 0) {
                    
                    // FIX? Sets -99 (not ideal) to a missing value and also marks it as missing.
                    if (colData == missingValue) {
                        colData = "-99";
                        _missingValues.push_back({row, col});
                    }
                    
                    attr = _c2a[col];
                    _data.coeffRef(row, attr) = strtod(colData.c_str(), &ch);
                    
                } else if (_colIsCatAttribute[col] && colData[0] != 0) {
                    
                    if (colData == missingValue) {
                        colData = "N/A";
                        _missingCatValues.push_back({row, col});
                    }
                    
                    attr = _c2ca[col];
                    if (std::find(_catItoS[attr].begin(), _catItoS[attr].end(), colData) == _catItoS[attr].end()) {
                        _catStoI[attr].insert(std::pair<std::string, int>(colData, (int)_catItoS[attr].size()));
                        _catItoS[attr].push_back(colData);
                    }
                    _catData.coeffRef(row, attr) = _catStoI[attr][colData];
                    
                }
                
                ++col;
                
            }
            
            ++row;
            
        } else break;
        
    }
    
    dataFile.close();
    
    for (unsigned a = 0; a < _nCatAttributes; ++a)
        _nCategories[a] = _catItoS[a].size();
    
}

std::vector<Eigen::MatrixXd *> metis::DataContainer::createPerClassMatrices(Eigen::MatrixXd *data,
                                                                            Eigen::ArrayXi *target) {
    
    unsigned nInstances = data->rows();
    unsigned nAttributes = data->cols();
    
    // Number of classes is initially unknown and is determined by reading target data.
    // During that process, indices of values contained on each class are also kept on a vector.
    unsigned nClasses = 0;
    std::vector<std::vector<unsigned>> exampleInClass;
    for (unsigned i = 0; i < nInstances; ++i) {
        
        // If there are more classes than initially estimated, vector is expanded.
        if (target->coeff(i) >= nClasses) {
            nClasses = target->coeff(i) + 1;
            unsigned vectorExpansion = nClasses - exampleInClass.size();
            for (unsigned x = 0; x < vectorExpansion; ++x) {
                exampleInClass.push_back({});
            }
        }
        
        exampleInClass[target->coeff(i)].push_back(i);
        
    }
    
    // A number of data sets equal to the number of classes is created.
    std::vector<Eigen::MatrixXd *> dividedData;
    
    for (unsigned c = 0; c < nClasses; ++c) {
        dividedData.push_back(new Eigen::MatrixXd(exampleInClass[c].size(), nAttributes));
        for (unsigned i = 0; i < exampleInClass[c].size(); ++i) {
            dividedData.back()->row(i) = data->row(exampleInClass[c][i]);
        }
    }
    
    return dividedData;
    
}

Eigen::ArrayXi metis::DataContainer::findNumberOfCategories(Eigen::MatrixXi *data) {
    
    unsigned nAttributes = data->cols();
    unsigned nInstances = data->rows();
    
    Eigen::ArrayXi nCategories(nAttributes);
    nCategories.setZero();
    
    for (unsigned a = 0; a < nAttributes; ++a) {
        for (unsigned i = 0; i < nInstances; ++i) {
            if (data->coeff(i, a) >= nCategories.coeff(a)) {
                nCategories.coeffRef(a) = data->coeff(i, a) + 1;
            }
        }
    }
    
    return nCategories;
    
}

metis::DataContainer::DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                              std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                              std::vector<std::vector<std::string>> catItoS,
                              std::vector<std::map<std::string, int>> catStoI) {
    
    std::ifstream dataFile;
    dataFile.open(filePath);
    
    if (!dataFile) {
        std::cerr << "Unable to open file." << std::endl;
        exit(1);
    }
    
    // Read one row and see how many columns it has on assumption that all
    // rows have the same number of columns. Then count number of all rows.
    std::string rowData, columnData;
    
    getline(dataFile, rowData);
    unsigned nRows = 1;
    unsigned nCols = 0;
    
    std::istringstream unscrambler(rowData);
    while (getline(unscrambler, columnData, separatorChar))
        if (columnData[0] != 0) ++nCols;
    
    while (getline(dataFile, rowData)) ++nRows;
    
    dataFile.close();
    
    _nInstances = nRows;
    _nAttributes = attributes.size();
    _nCatAttributes = catAttributes.size();
    _nCategories.resize(_nCatAttributes);
    
    if (catItoS.size() != 0) _catItoS = catItoS;
    else _catItoS.resize(_nCatAttributes);
    
    if (catStoI.size() != 0) _catStoI = catStoI;
    else _catStoI.resize(_nCatAttributes);
    
    _data.resize(_nInstances, _nAttributes);
    _catData.resize(_nInstances, _nCatAttributes);
    
    // Correlating ArrayXX columns with file attributes.
    _colIsAttribute.resize(nCols);
    _colIsCatAttribute.resize(nCols);
    for (unsigned c = 0; c < nCols; ++c) {
        _colIsAttribute[c] = false;
        _colIsCatAttribute[c] = false;
    }
    
    for (unsigned a = 0; a < attributes.size(); ++a) {
        _colIsAttribute[attributes[a]] = true;
        _c2a.insert(std::pair<unsigned, unsigned>(attributes[a], a));
    }
    
    for (unsigned a = 0; a < catAttributes.size(); ++a) {
        _colIsCatAttribute[catAttributes[a]] = true;
        _c2ca.insert(std::pair<unsigned, unsigned>(catAttributes[a], a));
    }
    
    readFile(filePath, separatorChar, missingValue, attributes, catAttributes, nRows, nCols);
    
}

metis::DataContainer::DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                              std::vector<unsigned> attributes, std::vector<unsigned> catAttributes) :
        DataContainer(filePath, separatorChar, missingValue, attributes, catAttributes, {}, {}) {
    
}

metis::DataContainer::~DataContainer() {

}
