//
// Copyright (c) 2018 Andreas Bampouris
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
    
    // Data
    //! Numerical data contained.
    Eigen::ArrayXXd _data;
    
    //! Categorical data contained
    Eigen::ArrayXXi _catData;
    
    // Meta-data
    unsigned _nInstances;
    
    unsigned _nAttributes;
    std::vector<bool> _colIsAttribute;
    //! Column of original file associated with respective column on ArrayXXd.
    std::map<unsigned, unsigned> _c2a;
    
    unsigned _nCatAttributes;
    std::vector<bool> _colIsCatAttribute;
    //! Column of original file associated with respective column on ArrayXXi.
    std::map<unsigned, unsigned> _c2ca;
    std::vector<std::vector<std::string>> _catItoS;
    std::vector<std::map<std::string, int>> _catStoI;
    //! Number of possible classes for each categorical attribute.
    std::vector<unsigned> _nCategories;
    
    //! Row and column of missing values.
    std::vector<std::array<unsigned, 2>> _missingValues;
    std::vector<std::array<unsigned, 2>> _missingCatValues;
    
    // Helper fuctions
    void readFile(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                  unsigned nRows, unsigned nCols);

public:
    
    // Access
    /**
     * Creates a matrix of continuous numerical values containing all data of certain columns.
     *
     * @param columns C++ vector with the indices of the columns relevant to the imminent data set.
     * @return Data set represented as a MatrixXd. Each row represents an instance, each column an attribute.
     */
    Eigen::MatrixXd createNumericalMatrix(std::vector<unsigned> columns);
    
    /**
     * Creates a matrix of integer values containing all data of certain columns.
     *
     * @param columns C++ vector with the indices of the columns relevant to the imminent data set.
     * @return Data set represented as a MatrixXi. Each row represents an instance, each column an attribute.
     */
    Eigen::MatrixXi createCategoricalMatrix(std::vector<unsigned> columns);
    
    /**
     * Creates an 1-D array of integer values containing data of a certain column.
     *
     * @param column Index of the relevant column.
     * @return Data set represented as an ArrayXi. Each row represents one instance's value for the relevant column.
     */
    Eigen::ArrayXi createClassArray(unsigned column);
    
    /**
     * Creates a matrix of binary values for the class represented on one column.
     *
     * @param column Index of the relevant column.
     * @return Data set represented as a MatrixXd. Each row represents an instance.
     *         If column is 1, said instance belongs on the respective (according to index) class, 0 otherwise.
     */
    Eigen::MatrixXd createBinaryMatrix(unsigned column);
    
    /**
     * Creates a matrix of binary values based on a 1-D array of integer values.
     *
     * @param data Data set represented as an ArrayXi. Each row represents one instance's value for the relevant column.
     * @return Data set represented as a MatrixXd. Each row represents an instance.
     *         If column is 1, said instance belongs on the respective (according to index) class, 0 otherwise.
     */
    static Eigen::MatrixXd convertToBinaryMatrix(Eigen::ArrayXi *data);
    
    /**
     * Creates matrices of input data each one of which consists of instances belonging to one class.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance.
     * @param target Data set represented as an ArrayXi. Each row represents an instance. Same size as data.
     * @return C++ vector of pointers to MatrixXd objects, each one of which is a matrix of input data of one class.
     */
    static std::vector<Eigen::MatrixXd *> createPerClassMatrices(Eigen::MatrixXd *data, Eigen::ArrayXi *target);
    
    /**
     * Determines the number of categories for each categorical attribute in data.
     *
     * @param data Data set represented as a MatrixXd. Each row represents an instance
     * @return ArrayXi with number of categories on each attribute.
     */
    static Eigen::ArrayXi findNumberOfCategories(Eigen::MatrixXi *data);
    
    void print() const;
    
    // Transformations
    void shuffle(Eigen::VectorXi order);
    void shuffle();
    
    /**
     * Deletes all data not between given indices.
     *
     * @param from Starting index.
     * @param to Terminal index.
     */
    void shrink(unsigned from, unsigned to);
    
    /**
     * Splits data between present DataContainer and a new one.
     *
     * @param proportion Proportion of data to be contained on new DataContainer.
     * @return New object of DataContainer class containing proportion of data.
     */
    DataContainer split(double proportion);
    
    /**
     * Mathematic transformations on data contained on present DataContainer.
     *
     * @param columns Indices of columns whose data will be transformed.
     */
    void standardize(std::vector<unsigned> columns);
    void rescale(std::vector<unsigned> columns);
    void logTransform(std::vector<unsigned> columns);
    void expTransform(std::vector<unsigned> columns);
    
    // Construction
    /**
     * Creates new DataContainer.
     *
     * @param data Numerical data contained.
     * @param catData Categorical data contained.
     * @param catItoS Names of each class on each categorical attribute.
     * @param catStoI Integer numbers corresponding to each class on each categorical attribute.
     */
    DataContainer(Eigen::ArrayXd data, Eigen::ArrayXi catData,
                  std::vector<std::vector<std::string>> catItoS, std::vector<std::map<std::string, int>> catStoI);
    
    /**
     * Creates new DataContainer based on file.
     *
     * @param filePath Location of file.
     * @param separatorChar Character separating the columns of each row.
     * @param missingValue String signaling missing values on file.
     * @param attributes Columns of file containing numerical data to be loaded.
     * @param catAttributes Columns of file containing categorical data to be loaded.
     * @param catItoS Names of each class on each categorical attribute.
     * @param catStoI Integer numbers corresponding to each class on each categorical attribute.
     */
    DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes,
                  std::vector<std::vector<std::string>> catItoS, std::vector<std::map<std::string, int>> catStoI);
    DataContainer(std::string &filePath, char separatorChar, std::string &missingValue,
                  std::vector<unsigned> attributes, std::vector<unsigned> catAttributes);
    
    ~DataContainer();
    
};

}


#endif //METIS_DATACONTAINER_H
