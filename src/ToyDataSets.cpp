//
// Copyright (c) 2018 Andreas Bampouris
//


#include "../include/ToyDataSets.h"

metis::DataLabeled *metis::loadIris() {

    std::string filePath = "../data/iris.data.txt";

    metis::DataSet *input = new metis::DataSet(filePath, ',');
    input->create({0,1,2,3});
    input->applyStandardization();

    metis::DataSet *output = new metis::DataSet(filePath, ',');
    output->create({4}, {4});

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataLabeled *metis::loadBanknotes() {

    std::string filePath = "../data/data_banknote_authentication.txt";

    metis::DataSet *input = new metis::DataSet(filePath, ',');
    input->create({0,1,2,3});
//    input->applyStandardization();

    metis::DataSet *output = new metis::DataSet(filePath, ',');
    output->create({4}, {4});

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataLabeled *metis::loadDiabetes() {

    std::string filePath = "../data/diabetes.tab.txt";

    metis::DataSet *input = new metis::DataSet(filePath, '\t');
    input->create({0,1,2,3,4,5,6,7,8,9});
    input->applyStandardization();
    input->scaleSquaredLengthToOne();

    metis::DataSet *output = new metis::DataSet(filePath, '\t');
    output->create({10,11});

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataLabeled *metis::loadWine() {

    std::string filePath = "../data/wine.data.txt";

    metis::DataSet *input = new metis::DataSet(filePath, ',');
    input->create({1,2,3,4,5,6,7,8,9,10,11,12,13});
    input->applyStandardization();

    metis::DataSet *output = new metis::DataSet(filePath, ',');
    output->create({0}, {0});

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataLabeled *metis::loadMNIST(bool testSet) {

    std::string filePath;
    if (!testSet) filePath = "../data/mnist_train.csv";
    else filePath = "../data/mnist_test.csv";

    metis::DataSet *input = new metis::DataSet(filePath, ',');
    std::vector<unsigned> attr;
    for (unsigned a = 1; a < 785; ++a) attr.push_back(a);
    input->create(attr, {});
    input->applyStandardization();

    metis::DataSet *output = new metis::DataSet(filePath, ',');
    std::vector<std::map<std::string, unsigned>> dict(1);
    dict[0].insert(std::pair<std::string, unsigned>("0", 0));
    dict[0].insert(std::pair<std::string, unsigned>("1", 1));
    dict[0].insert(std::pair<std::string, unsigned>("2", 2));
    dict[0].insert(std::pair<std::string, unsigned>("3", 3));
    dict[0].insert(std::pair<std::string, unsigned>("4", 4));
    dict[0].insert(std::pair<std::string, unsigned>("5", 5));
    dict[0].insert(std::pair<std::string, unsigned>("6", 6));
    dict[0].insert(std::pair<std::string, unsigned>("7", 7));
    dict[0].insert(std::pair<std::string, unsigned>("8", 8));
    dict[0].insert(std::pair<std::string, unsigned>("9", 9));
    output->create({0}, {0}, dict);

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataLabeled *metis::loadMNIST() {
    return loadMNIST(false);
}

metis::DataLabeled *metis::loadBC() {

    std::string filePath = "../data/breast-cancer.data.txt";

    metis::DataSet *input = new metis::DataSet(filePath, ',');
    input->create({1,2,3,4,5,6,7,8,9}, {1,2,3,4,5,6,7,8,9});

    metis::DataSet *output = new metis::DataSet(filePath, ',');
    output->create({0}, {0});

    metis::DataLabeled *labeledData = new metis::DataLabeled(input, output);

    return labeledData;

}

metis::DataContainer *metis::loadIrisContainer() {
    
    std::string filePath = "../data/iris.data.txt";
    std::string missingValue = "?";
    
    metis::DataContainer *data = new metis::DataContainer(filePath, ',', missingValue, {0,1,2,3}, {4});
    data->standardize({0,1,2,3});
    
    return data;
    
}

metis::DataContainer *metis::loadBanknotesContainer() {

    std::string filePath = "../data/data_banknote_authentication.txt";
    std::string missingValue = "?";

    metis::DataContainer *data = new metis::DataContainer(filePath, ',', missingValue, {0,1,2,3}, {4});
//    data->standardize({0,1,2,3});

    return data;

}

metis::DataContainer *metis::loadDiabetesContainer() {
    
    std::string filePath = "../data/diabetes.tab.txt";
    std::string missingValue = "?";
    
    metis::DataContainer *data =
            new metis::DataContainer(filePath, '\t', missingValue, {0,1,2,3,4,5,6,7,8,9,10,11}, {});
    data->standardize({0,1,2,3,4,5,6,7,8,9});
    data->rescale({0,1,2,3,4,5,6,7,8,9});
    
    return data;
    
}

metis::DataContainer *metis::loadWineContainer() {
    
    std::string filePath = "../data/wine.data.txt";
    std::string missingValue = "?";
    
    metis::DataContainer *data =
            new metis::DataContainer(filePath, ',', missingValue, {1,2,3,4,5,6,7,8,9,10,11,12,13}, {0});
    data->standardize({1,2,3,4,5,6,7,8,9,10,11,12,13});
    
    return data;
    
}

metis::DataContainer *metis::loadMNISTContainer(bool testSet) {
    
    std::string filePath;
    if (!testSet) filePath = "../data/mnist_train.csv";
    else filePath = "../data/mnist_test.csv";
    std::string missingValue = "?";
    
    std::vector<unsigned> attr;
    for (unsigned a = 1; a < 785; ++a) attr.push_back(a);
    
    std::vector<std::map<std::string, int>> catStoI(1);
    catStoI[0].insert(std::pair<std::string, int>("0", 0));
    catStoI[0].insert(std::pair<std::string, int>("1", 1));
    catStoI[0].insert(std::pair<std::string, int>("2", 2));
    catStoI[0].insert(std::pair<std::string, int>("3", 3));
    catStoI[0].insert(std::pair<std::string, int>("4", 4));
    catStoI[0].insert(std::pair<std::string, int>("5", 5));
    catStoI[0].insert(std::pair<std::string, int>("6", 6));
    catStoI[0].insert(std::pair<std::string, int>("7", 7));
    catStoI[0].insert(std::pair<std::string, int>("8", 8));
    catStoI[0].insert(std::pair<std::string, int>("9", 9));
    
    std::vector<std::vector<std::string>> catItoS(1);
    catItoS[0].push_back("0");
    catItoS[0].push_back("1");
    catItoS[0].push_back("2");
    catItoS[0].push_back("3");
    catItoS[0].push_back("4");
    catItoS[0].push_back("5");
    catItoS[0].push_back("6");
    catItoS[0].push_back("7");
    catItoS[0].push_back("8");
    catItoS[0].push_back("9");
    
    metis::DataContainer *data = new metis::DataContainer(filePath, ',', missingValue, attr, {0}, catItoS, catStoI);
    data->standardize(attr);
    
    return data;
    
}

metis::DataContainer *metis::loadMNISTContainer() {
    return loadMNISTContainer(false);
}

metis::DataContainer *metis::loadBCContainer() {
    
    std::string filePath = "../data/breast-cancer.data.txt";
    std::string missingValue = "?";
    
    metis::DataContainer *data = new metis::DataContainer(filePath, ',', missingValue, {}, {0,1,2,3,4,5,6,7,8,9});
    
    return data;
    
}
