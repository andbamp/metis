//
// Copyright (c) 2018 Andreas Bampouris
//

#include <fstream>
#include "../include/DataLabeled.h"

void metis::DataLabeled::setInput(metis::DataSet *input) {
    _input = input;
}

void metis::DataLabeled::setOutput(metis::DataSet *output) {
    _output = output;
}

void metis::DataLabeled::shuffle() {

    unsigned instances = _input->instances();

    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(instances, 0, instances);
    std::random_shuffle(indices.data(), indices.data() + instances);

    _input->shuffle(indices);
    _output->shuffle(indices);

}

metis::DataLabeled *metis::DataLabeled::split(double proportion) {

    double cut = instances() * (1 - proportion) + .5;
    unsigned newSize = (unsigned) cut;

    DataSet *splitInput = new DataSet();
    *splitInput = *_input;
    splitInput->shrink(newSize, instances());

    DataSet *splitOutput = new DataSet();
    *splitOutput = *_output;
    splitOutput->shrink(newSize, instances());

    DataLabeled *split = new DataLabeled();
    split->setInput(splitInput);
    split->setOutput(splitOutput);

    _input->shrink(0, newSize);
    _output->shrink(0, newSize);

    return split;

}

unsigned metis::DataLabeled::instances() const {
    return _input->instances();
}

unsigned metis::DataLabeled::inputs() const {
    return _input->attributes();
}

unsigned metis::DataLabeled::outputs() const {
    return _output->attributes();
}

unsigned metis::DataLabeled::categories(unsigned attribute) const {
    return _input->categories(attribute);
}

unsigned metis::DataLabeled::classes(unsigned attribute) const {
    return _output->categories(attribute);
}

metis::DataSet *metis::DataLabeled::getInputs() const {
    return _input;
}

metis::DataSet *metis::DataLabeled::getOutputs() const {
    return _output;
}

void metis::DataLabeled::producePlotInstructions(std::vector<unsigned> inputs, std::vector<unsigned> outputs) const {

    // Create data file
    Eigen::MatrixXd *inData = _input->getData();
    Eigen::MatrixXd *outData = _output->getData();

    std::ofstream fileData;
    fileData.open("points.txt");

    for (unsigned i = 0; i < instances(); ++i) {

        for (unsigned a = 0; a < _input->attributes(); ++a) {
            fileData << inData->coeffRef(i, a) << " " << std::flush;
        }

        for (unsigned a = 0; a < _output->attributes(); ++a) {
            fileData << outData->coeffRef(i, a) << " " << std::flush;
        }

        fileData << std::endl;

    }

    fileData.close();

    // Create instructions file
    std::ofstream fileInstr;
    fileInstr.open("instructions.txt");

    unsigned nIn = _input->attributes();

    if (inputs.size() == 1 && outputs.size() == 1) {

        fileInstr << "plot \"points.txt\" u " << std::flush;
        fileInstr << inputs[0]+1 << ":" << std::flush;
        fileInstr << nIn+outputs[0]+1 << std::endl;

    } else if (inputs.size() == 2 && outputs.size() == 1) {

        fileInstr << "plot \"points.txt\" u " << std::flush;
        fileInstr << inputs[0]+1 << ":" << std::flush;
        fileInstr << inputs[1]+1 << ":" << std::flush;
        fileInstr << nIn+outputs[0]+1 << " " << std::flush;
        fileInstr << "with points palette" << std::endl;

    } else {

        std::cout << "Wrong parameters given for plot." << std::endl;
        fileInstr.clear();
        fileInstr.close();
        exit(1);

    }

    fileInstr.close();

}

metis::DataLabeled::DataLabeled(metis::DataSet *input, metis::DataSet *output) {
    _input = input;
    _output = output;
}

metis::DataLabeled::DataLabeled(metis::DataSet *input, Eigen::VectorXi labels) {
    _input = input;
    _output = new DataSet(labels.cast<double>());
}

metis::DataLabeled::DataLabeled() {

}

metis::DataLabeled::~DataLabeled() {

    delete _input;
    delete _output;

}
