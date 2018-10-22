//
// Copyright (c) 2018 Andreas Bampouris
//

#ifndef METIS_PREDICTOR_H
#define METIS_PREDICTOR_H


namespace metis {

template <class I, class T>
class Predictor {

protected:

    // Meta-data
    
    //! Number of input attributes.
    unsigned _nAttributes;

public:

    // Access
    
    /**
     * General method for fitting a model to a certain set of labeled data.
     *
     * @param input Input data of training set.
     * @param target Output data of training set.
     * @param valInput Input data of validation set.
     * @param valTarget Output data of validation set.
     * @param verboseCycle Number of updates before user is informed for training state.
     */
    virtual void fit(I *input, T *target, I *valInput, T *valTarget, unsigned verboseCycle) = 0;
    virtual void fit(I *input, T *target, unsigned verboseCycle);
    virtual void fit(I *input, T *target, I *valInput, T *valTarget);
    virtual void fit(I *input, T *target);
    
    /**
     * General method for predicting output for a given input.
     *
     * @param input Input data with unknown output.
     */
    virtual T predict(I *input) const = 0;
    
    /**
     * General method for assessing correctness.
     *
     * @param input Input data of test set.
     * @param target Output data of test set.
     */
    virtual double score(I *input, T *target) const = 0;

};

}


#endif //METIS_PREDICTOR_H
