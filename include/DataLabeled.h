//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_DATALABELED_H
#define METIS_DATALABELED_H


#include "DataSet.h"

namespace metis {

class DataLabeled {

private:

    // 3. Structure
    DataSet *_input = nullptr;
    DataSet *_output = nullptr;

    // 4. State

    // 5. Helper methods
    void produceRegressionPlotInstructions(unsigned x, unsigned y) const;
    void produceClusteringPlotInstructions(unsigned x, unsigned y, unsigned z) const;

public:

    // 2. Interface methods
    void setInput(DataSet *input);
    void setOutput(DataSet *output);

    void shuffle();
    DataLabeled *split(double proportion);

    unsigned instances() const;
    unsigned inputs() const;
    unsigned outputs() const;

    unsigned categories(unsigned attribute) const;
    unsigned classes(unsigned attribute) const;

    DataSet *getInputs() const;
    DataSet *getOutputs() const;

    void producePlotInstructions(std::vector<unsigned> inputs, std::vector<unsigned> outputs) const;

    // 1. Construction
    DataLabeled(DataSet *input, DataSet *output);
    DataLabeled(DataSet *input, Eigen::VectorXi labels);
    DataLabeled();
    ~DataLabeled();

};

}


#endif //METIS_DATALABELED_H
