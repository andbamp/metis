//
// Created by Andreas Bampouris on 11/10/2018.
//

#ifndef METIS_DATACONTAINER_H
#define METIS_DATACONTAINER_H


#include <Eigen/Dense>
#include <vector>
#include <array>
#include <map>

namespace metis {

class DataContainer {

private:
    
    // 3. Data
    Eigen::ArrayXXd _data;
    Eigen::ArrayXXi _catData;
    
    // 4. Meta-data
    unsigned _nInstances;
    
    unsigned _nAttributes;
    std::vector<bool> _colIsAttribute;
    std::map<unsigned, unsigned> _c2a;
    
    unsigned _nCatAttributes;
    std::vector<bool> _colIsCatAttribute;
    std::map<unsigned, unsigned> _c2ca;
    std::vector<std::vector<std::string>> _catItoS;
    std::vector<std::map<std::string, int>> _catStoI;
    std::vector<unsigned> _nCategories;
    
    std::vector<std::array<unsigned, 2>> _missingValues;
    std::vector<std::array<unsigned, 2>> _missingCatValues;
    
    // 5. Helper fuctions
    void readFile(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                  unsigned nRows, unsigned nCols);

public:
    
    // 2. Access
    Eigen::MatrixXd createNumericalMatrix(std::vector<unsigned> columns);
    Eigen::MatrixXi createCategoricalMatrix(std::vector<unsigned> columns);
    Eigen::ArrayXi createClassArray(unsigned column);
    Eigen::MatrixXd createBinaryMatrix(unsigned column);
    
    void print() const;
    
    // 2b. Transformations
    void shuffle(Eigen::VectorXi order);
    void shuffle();
    
    void shrink(unsigned from, unsigned to);
    DataContainer split(double proportion);
    
    void standardize(std::vector<unsigned> columns);
    void rescale(std::vector<unsigned> columns);
    void logTransform(std::vector<unsigned> columns);
    void expTransform(std::vector<unsigned> columns);
    
    // 2c. Getters
    
    
    // 1. Construction
    DataContainer(Eigen::ArrayXd data, Eigen::ArrayXi catData,
                  std::vector<std::vector<std::string>> catItoS, std::vector<std::map<std::string, int>> catStoI);
    DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                  std::vector<std::vector<std::string>> catItoS, std::vector<std::map<std::string, int>> catStoI);
    DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes);
    ~DataContainer();
    
};

}


#endif //METIS_DATACONTAINER_H
