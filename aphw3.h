#ifndef _APHW_H_
#define _APHW_H_

#include <iostream>
#include <string>
#include <fstream>
#include "dataset.h"
#include "Result.h"
#include "NeuralNet.h"

Dataset loadFuncDataset(const char* filename);
std::vector<Result> testNeuralNets(Dataset& dataset, std::vector<size_t>& hidden_neurons, 
                                    double lr=0.01, size_t max_iters = 10000, 
                                    const char* af1 = "Sigmoid", const char* af2 = "Linear");

Result findBestNeuralNet(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr = 0.01, 
                                    size_t max_iters = 10000, const char* af1 = "Sigmoid", 
                                    const char* af2 = "Linear");

void estimateFunction(const char* filename, size_t hidden_neurons_no);
void estimateFunction(Dataset, size_t hidden_neurons_no); //mine!


#endif
