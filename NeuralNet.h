#ifndef _NEURAL_H_
#define _NEURAL_H_

#include <iostream>
#include "Matrix.h"
#include "dataset.h"
#include <math.h>
#include <string.h>
#include "Result.h"


class NeuralNet
{
private:
    Matrix w1; // Weights of layer 1
    Matrix w2; // Weights of layer 2
    Matrix b1; // Biases of layer 1
    Matrix b2; // Biases of layer 2
    Matrix a1; // Output of layer 1
    Matrix a2; // Output of layer 2
    Matrix n1; // n vector for layer 1
    Matrix n2; // n vector for layer 2
    Matrix s1; // s vector for layer 1
    Matrix s2; // s vector for layer 2
    const char* af1{"Sigmoid"};
    const char* af2{"Linear"};  //or "Linear"
    size_t hidden_layer_neurons{3};
    double lr{0.01}; //learning rate
    size_t max_iters{1000}; //avazesh kardam!
    double min_loss{0.01};
    Dataset dataset;

public:
    NeuralNet(Dataset dataset, size_t hidden_layer_neurons,    //make changes here!
                                const char* f1 = "Sigmoid", 
                                const char* f2 = "Linear", 
                                double lr = 0.01, 
                                size_t max_iters = 1000, 
                                double min_loss = 0.01);
    Matrix forwardPropagate(Matrix& input);
    void backPropagate(Matrix& input, Matrix& target);
    double trainLoss();
    double testLoss();

    void setW1(Matrix& w);
    void setW2(Matrix& w);
    void setB1(Matrix& b);
    void setB2(Matrix& b);
    Matrix getW1();
    Matrix getW2();
    Matrix getB1();
    Matrix getB2();

    Result fit();
    std::vector<double> testResult(); //mine!
    void show();
    const char* show2();
    
    friend std::ostream &operator<<(std::ostream &os, NeuralNet &nn);
};

Matrix sigmoid(Matrix);
double Sigmoid(double);
Matrix linear(Matrix);
Matrix sigmoidDot(size_t number_of_neurons, Matrix n);
Matrix linearDot(size_t number_of_neurons, Matrix n);


#endif