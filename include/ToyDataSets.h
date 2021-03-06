//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_TOYDATASETS_H
#define METIS_TOYDATASETS_H

#include "DataLabeled.h"

namespace metis {

DataLabeled *loadIris();
DataLabeled *loadBanknotes();
DataLabeled *loadDiabetes();
DataLabeled *loadWine();
DataLabeled *loadMNIST(bool testSet);
DataLabeled *loadMNIST();
DataLabeled *loadBC();

}

#endif //METIS_TOYDATASETS_H
