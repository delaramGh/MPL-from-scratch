#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <array>
#include "Matrix.h"

class Dataset
{
private:
    Matrix inputs;
    Matrix targets;
    Matrix train_inputs;
    Matrix train_targets;
    Matrix test_inputs;
    Matrix test_targets;
    double percentage{70};
    size_t no_of_samples;
    size_t input_dim;
    size_t target_dim;
	
public:
    Dataset(Matrix inputs, Matrix targets, double percentage=70);
    Dataset(Matrix inputs, Matrix targets, Matrix train_inputs, 
                        Matrix train_targets, Matrix test_inputs, Matrix test_targets);
    Dataset();     //default constructor
    size_t getNoOfSamples();
    size_t getNoOfTrainSamples();
    size_t getNoOfTestSamples();
    size_t getInputDim();
    size_t getTargetDim();
    Matrix getInputs();
    Matrix getTargets();
    Matrix getTrainInputs();
    Matrix getTrainTargets();
    Matrix getTestInputs();
    Matrix getTestTargets();
    void shuffle(); 
    void show();
    size_t show1();
    std::vector<Matrix> operator [] (size_t i);
    Dataset operator+(const Dataset& dataset);

    //friend std::ostream &operator<<(std::ostream &os, Dataset &d); 
};

std::ostream &operator<<(std::ostream &os, Dataset &d);

#endif




