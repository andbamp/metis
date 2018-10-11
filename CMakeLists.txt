cmake_minimum_required(VERSION 3.10)
project(metis)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(metis src/DataSet.cpp include/DataSet.h src/DataLabeled.cpp include/DataLabeled.h include/ToyDataSets.h src/ToyDataSets.cpp src/KMeans.cpp include/KMeans.h src/LogisticRegression.cpp include/LogisticRegression.h src/MLPClassifier.cpp include/MLPClassifier.h demos/DemosMetis.h include/NaiveBayes.h src/GaussianNB.cpp include/GaussianNB.h src/MultinomialNB.cpp include/MultinomialNB.h src/NaiveBayes.cpp src/LinearRegression.cpp include/LinearRegression.h src/DecisionTree.cpp include/DecisionTree.h src/DTNode.cpp include/DTNode.h src/core/Predictor.cpp src/core/Predictor.h src/core/Regressor.cpp src/core/Regressor.h src/core/Classifier.cpp src/core/Classifier.h src/DataContainer.cpp include/DataContainer.h src/core/Optimizer.cpp src/core/Optimizer.h src/LinearRegressionNew.cpp include/LinearRegressionNew.h)

target_link_libraries(metis Eigen3::Eigen)


add_executable(libtest main.cpp demos/KMeansDemoIris.h demos/LinRegDemo.h demos/LogRegDemoMNIST.h demos/KMeansDemoFrogs.h demos/MLPDemoIris.h demos/MLPDemoMNIST.h demos/NBMultinomialDemo.h demos/NBGaussianDemo.h demos/LogRegDemoIris.h demos/LogRegDemoBanknotes.h demos/DTreeDemo.h)

target_link_libraries(libtest metis)



#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -march=native -fopenmp")