#ifndef _RESULT_H
#define _RESULT_H

#include <iostream>
#include <iomanip>

class Result
{
private:
    double train_loss;
    double test_loss;
    size_t no_of_hidden_neurons;
    double lr{0.01};
    size_t iters{10000};
    const char* af1 = "Sigmoid";
    const char* af2 = "Linear"; 

public:
    Result(double train_loss, double test_loss, size_t no_of_hidden_neurons, 
                                double lr = 0.01, size_t iters = 10000, 
                                const char* af1 = "Sigmoid", const char* af2 = "Linear");
    explicit Result(double test_loss);

    bool operator < (Result);
    bool operator > (Result);
    bool operator <= (Result);
    bool operator >= (Result);
    double getTestLoss();
    void show();
    const char* show2();

    friend std::ostream &operator<<(std::ostream &os, Result &r);
};


#endif


