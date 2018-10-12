//
// Copyright (c) 2018 Andreas Bampouris
//

#include "Optimizer.h"


void metis::Optimizer::transferFunction(double *prediction, unsigned transferFunc) {

    switch (transferFunc) {
    
        case 0:
            // If transferFunc == 0 the prediction is equal to the linear transformation that was given as input.
            break;
        
        case 1:
            // If transferFunc == 1 the prediction is calculated as the sigmoid function of the linear transformation.
            *prediction *= -1;
            *prediction = std::exp(*prediction);
            *prediction += 1.0;
            *prediction = 1 / *prediction;
            break;
        
        default:
            break;
    
    }

}

void metis::Optimizer::transferFunction(Eigen::VectorXd *prediction, unsigned transferFunc) {
    
    switch (transferFunc) {
        
        case 0:
            // If transferFunc == 0 the prediction is equal to the linear transformation that was given as input.
            break;
        
        case 1:
            // If transferFunc == 1 the prediction is calculated as the sigmoid function of the linear transformation.
            prediction->array() *= -1;
            prediction->array() = prediction->array().exp();
            prediction->array() += 1.0;
            prediction->array() = prediction->array().inverse();
            break;
        
        default:
            break;
        
    }

}

void metis::Optimizer::batchGradientDescent(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
                                            unsigned transferFunc) {
    
    if (_batchSize == 0) _batchSize = input->rows();
    unsigned nBatches = input->rows() / _batchSize;
    
    for (unsigned c = 0; c < _nModels; ++c) {
        
        Eigen::VectorXd w = _coeff.transpose().col(c);
        Eigen::VectorXd dw(_nModels);
        double b = _intercept.coeff(c);
        double db;
        Eigen::VectorXd probability(_batchSize);
        
        for (unsigned i = 0; i < _iterations; ++i) {
            
            for (unsigned r = 0; r < nBatches; ++r) {
    
                // Linear transformation is calculated for a mini-batch of examples.
                probability = input->block(r * _batchSize, 0, _batchSize, _nModels) * w;
                probability.array() += b;
    
                // The actual predictions are found through (possibly) a non-linear transformation.
                transferFunction(&probability, transferFunc);
    
                // Errors are calculated and parameters are corrected.
                probability -= target->block(r * _batchSize, c, _batchSize, 1);
                dw.array() = probability.transpose() * input->block(r * _batchSize, 0, _batchSize, _nModels);
                dw.array() /= _batchSize;
                db = probability.sum() / _batchSize;
                
                w -= _learnRate * dw;
                b -= _learnRate * db;
                
            }
            
        }
        
        _coeff.row(c) = w.transpose();
        _intercept.coeffRef(c) = b;
        
    }
    
}

void metis::Optimizer::stochasticGradientDescent(Eigen::MatrixXd *input, Eigen::MatrixXd *target,
                                                 unsigned transferFunc) {
    
    unsigned nInstances = input->rows();
    
    for (unsigned c = 0; c < _nModels; ++c) {
        
        Eigen::VectorXd w = _coeff.transpose().col(c);
        Eigen::VectorXd dw(_nModels);
        double b = _intercept.coeff(c);
        double db;
        double probability;
        
        for (unsigned i = 0; i < _iterations; ++i) {
        
            for (unsigned r = 0; r < nInstances; ++r) {
            
                // Linear transformation is calculated for one example.
                probability = (input->row(r) * w).coeff(0);
                probability += b;
    
                // The actual prediction is found through (possibly) a non-linear transformation.
                transferFunction(&probability, transferFunc);
                
                // Error is calculated and parameters are corrected.
                probability -= target->coeff(r, c);
                dw.array() = probability * input->row(r);
                db = probability;
                
                w -= _learnRate * dw;
                b -= _learnRate * db;
            
            }
        
        }
        
        _coeff.row(c) = w.transpose();
        _intercept.coeffRef(c) = b;

    }
    
}

Eigen::MatrixXd metis::Optimizer::getCoefficients() {
    return _coeff;
}

Eigen::VectorXd metis::Optimizer::getIntercepts() {
    return _intercept;
}
